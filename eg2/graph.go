package eg2

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/daaku/eg2npu/gguf"
	"github.com/daaku/eg2npu/ir"
)

// Graph inputs and output. The token embeddings are looked up on the host, so
// the graph starts from the scaled embedding of each token.
const (
	InEmbeds  = "embeds"   // f32 [1, L, Embd]: token embedding times sqrt(Embd)
	InMask    = "mask"     // f32 [1, 1, 1, 1, L]: 0 for a token, maskValue for padding
	InMaskSWA = "mask_swa" // f32 [1, 1, 1, L, L]: InMask plus the sliding window
	InCos     = "cos"      // f32 [L, HeadDim/2] for the global layers
	InSin     = "sin"
	InCosSWA  = "cos_swa" // f32 [L, HeadDimSWA/2] for the sliding window layers
	InSinSWA  = "sin_swa"
	OutEmbeds = "embeddings" // f32 [1, L, EmbdOut], before pooling
)

// maskValue hides a key. It is finite and fits in float16, which is what the
// NPU computes in: an infinity there turns a softmax into NaNs more easily
// than it hides anything.
const maskValue = -30000

// Weights is how the IR stores the linear layers' weights.
type Weights string

const (
	// WeightsF16 dequantizes every weight to float16.
	WeightsF16 Weights = "f16"
	// WeightsQ8 keeps Q8_0 weights as int8 with a float16 scale per block of
	// 32, exactly the numbers llama.cpp multiplies by.
	WeightsQ8 Weights = "q8"
)

// BuildOptions choose the graph's variant.
type BuildOptions struct {
	// Length is the sequence length, or -1 for a dynamic one.
	Length  int64
	Weights Weights
	// MaxScores bounds the attention scores one block of queries makes (rows
	// times keys) in a static graph: longer sequences attend in blocks of
	// query rows. The NPU faulted at run time on a 1024 token sequence
	// attended whole, and ran 768. 0 attends whole.
	MaxScores int64
	// ResidualScale divides the residual stream, which RMS norms cannot see,
	// to keep it inside float16's range. 1 leaves the graph as llama.cpp's.
	ResidualScale float32
}

type builder struct {
	g   *ir.Graph
	f   *gguf.File
	cfg Config
	opt BuildOptions
}

// Build writes the encoder for one sequence length as an IR graph whose
// constants go into pool.
func Build(f *gguf.File, cfg Config, pool *ir.Pool, opt BuildOptions) (*ir.Graph, error) {
	if opt.ResidualScale == 0 {
		opt.ResidualScale = 1
	}
	if opt.Weights == "" {
		opt.Weights = WeightsF16
	}
	b := &builder{g: ir.New("embeddinggemma2", pool), f: f, cfg: cfg, opt: opt}
	if err := b.build(); err != nil {
		return nil, err
	}
	return b.g, b.g.Err
}

func (b *builder) tensor(name string) (*gguf.Tensor, error) { return b.f.Tensor(name) }

// vector is a 1D weight (a norm's), as float32, times scale.
func (b *builder) vector(name string, scale float32) (ir.Value, error) {
	t, err := b.tensor(name)
	if err != nil {
		return ir.Value{}, err
	}
	v := b.f.Float32(t)
	for i := range v {
		v[i] *= scale
	}
	return b.g.F32s([]int64{int64(len(v))}, v...), nil
}

// linear is x times the weight's transpose: the weight is stored {in, out} in
// ggml order, which is [out][in] row major, PyTorch's layout.
func (b *builder) linear(x ir.Value, name string) (ir.Value, error) {
	return b.linearRows(x, name, 0, -1)
}

// linearRows is linear with only the outputs [from, to) of the weight; a to
// of -1 is all of them.
func (b *builder) linearRows(x ir.Value, name string, from, to int) (ir.Value, error) {
	t, err := b.tensor(name)
	if err != nil {
		return ir.Value{}, err
	}
	in := t.Dims[0]
	if to < 0 {
		to = t.Dims[1]
	}
	out := to - from
	var w ir.Value
	if b.opt.Weights == WeightsQ8 && t.Type == gguf.Q8_0 {
		// int8 values [out, in/32, 32] times float16 scales [out, in/32, 1]:
		// the decompression pattern OpenVINO plugins keep compressed.
		raw := b.f.Raw(t)[from*in/32*34 : to*in/32*34]
		blocks := len(raw) / 34
		q := make([]byte, blocks*32)
		d := make([]byte, blocks*2)
		for i := range blocks {
			copy(d[i*2:], raw[i*34:i*34+2])
			copy(q[i*32:], raw[i*34+2:i*34+34])
		}
		qv := b.g.Convert(b.g.Const(ir.I8, []int64{int64(out), int64(in / 32), 32}, q), ir.F16)
		dv := b.g.Const(ir.F16, []int64{int64(out), int64(in / 32), 1}, d)
		w = b.g.Convert(b.g.Reshape(b.g.Mul(qv, dv), int64(out), int64(in)), ir.F32)
	} else {
		v := b.f.Float32(t)[from*in : to*in]
		h := make([]byte, 2*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint16(h[i*2:], gguf.FloatToHalf(x))
		}
		w = b.g.Convert(b.g.Const(ir.F16, []int64{int64(out), int64(in)}, h), ir.F32)
	}
	return b.g.MatMul(x, w, false, true), nil
}

// rms is llama.cpp's RMS norm over the last axis, times weight when there is
// one.
func (b *builder) rms(x ir.Value, weight string, scale float32) (ir.Value, error) {
	g := b.g
	sq := g.Pow(x, g.Scalar(2))
	norm := g.Div(x, g.Sqrt(g.Add(g.MeanLast(sq), g.Scalar(b.cfg.Eps))))
	if weight == "" {
		return norm, nil
	}
	w, err := b.vector(weight, scale)
	if err != nil {
		return ir.Value{}, err
	}
	return g.Mul(norm, w), nil
}

// rope rotates x [1, L, H, D] by NEOX's rule: the first half of each head
// pairs with the second.
func (b *builder) rope(x, cos, sin ir.Value, d int64) ir.Value {
	g := b.g
	x1 := g.Slice(x, -1, 0, d/2)
	x2 := g.Slice(x, -1, d/2, d)
	return g.Concat(-1,
		g.Sub(g.Mul(x1, cos), g.Mul(x2, sin)),
		g.Add(g.Mul(x2, cos), g.Mul(x1, sin)))
}

// attend is softmax(q k^T + mask) v, for q [1, KV, G, L, D] and k, v
// [1, KV, 1, L, D]. There is no 1/sqrt(D): the q and k norms make it
// unnecessary, and llama.cpp scales by 1.
//
// With a static length past MaxScores it works on blocks of query rows, and a
// block in a sliding window layer reads only the keys its window reaches.
func (b *builder) attend(q, k, v, mask ir.Value, swa bool) ir.Value {
	g := b.g
	L := b.opt.Length
	whole := func(q, k, v, mask ir.Value) ir.Value {
		return g.MatMul(g.Softmax(g.Add(g.MatMul(q, k, false, true), mask), -1), v, false, false)
	}
	if L < 0 || b.opt.MaxScores <= 0 || L*L <= b.opt.MaxScores {
		return whole(q, k, v, mask)
	}
	half := int64(b.cfg.Window / 2)
	block := int64(1)
	for {
		next := block * 2
		keys := L
		if swa {
			keys = min(L, next+2*half)
		}
		if next > L || next*keys > b.opt.MaxScores {
			break
		}
		block = next
	}
	var parts []ir.Value
	for start := int64(0); start < L; start += block {
		end := min(L, start+block)
		qb := g.Slice(q, 3, start, end)
		kb, vb, mb := k, v, mask
		if swa {
			// Rows [start, end) see keys [start-half, end+half].
			ks, ke := max(0, start-half), min(L, end+half)
			kb, vb = g.Slice(k, 3, ks, ke), g.Slice(v, 3, ks, ke)
			mb = g.Slice(g.Slice(mask, 3, start, end), 4, ks, ke)
		}
		parts = append(parts, whole(qb, kb, vb, mb))
	}
	return g.Concat(3, parts...)
}

func (b *builder) build() error {
	g, cfg, L := b.g, b.cfg, b.opt.Length
	s := b.opt.ResidualScale
	embd := int64(cfg.Embd)

	x := g.Parameter(InEmbeds, ir.F32, 1, L, embd)
	mask := g.Parameter(InMask, ir.F32, 1, 1, 1, 1, L)
	maskSWA := mask
	if L < 0 || cfg.NeedsWindow(int(L)) {
		maskSWA = g.Parameter(InMaskSWA, ir.F32, 1, 1, 1, L, L)
	}
	var dimFull, dimSWA int64
	for i := range cfg.Layers {
		if cfg.SWA[i] {
			dimSWA = int64(cfg.HeadDim[i])
		} else {
			dimFull = int64(cfg.HeadDim[i])
		}
	}
	var cos, sin, cosSWA, sinSWA ir.Value
	if dimFull > 0 {
		cos = g.Reshape(g.Parameter(InCos, ir.F32, L, dimFull/2), 1, -1, 1, dimFull/2)
		sin = g.Reshape(g.Parameter(InSin, ir.F32, L, dimFull/2), 1, -1, 1, dimFull/2)
	}
	if dimSWA > 0 {
		cosSWA = g.Reshape(g.Parameter(InCosSWA, ir.F32, L, dimSWA/2), 1, -1, 1, dimSWA/2)
		sinSWA = g.Reshape(g.Parameter(InSinSWA, ir.F32, L, dimSWA/2), 1, -1, 1, dimSWA/2)
	}

	// The per layer inputs come from the scaled embeddings alone:
	// norm(proj(x) / sqrt(Embd)), a PerLayer slice of it for each layer.
	// Each layer's slice is normalized on its own, so it is computed on its
	// own: whole, it is [1, L, Layers * PerLayer], 50 MB of float16 at 2048
	// tokens, and the NPU faulted running that.
	x0 := x
	perLayerInput := func(il int) (ir.Value, error) {
		p, err := b.linearRows(x0, "per_layer_model_proj.weight", il*cfg.PerLayer, (il+1)*cfg.PerLayer)
		if err != nil {
			return ir.Value{}, err
		}
		p = g.Mul(p, g.Scalar(float32(1/math.Sqrt(float64(cfg.Embd)))))
		return b.rms(p, "per_layer_proj_norm.weight", 1)
	}

	// The residual stream is carried divided by s. Every branch is added to
	// it through a norm whose weight absorbs the 1/s, and every reader of it
	// is a norm, which cannot tell.
	if s != 1 {
		x = g.Mul(x, g.Scalar(1/s))
	}
	for il := range cfg.Layers {
		blk := func(n string) string { return fmt.Sprintf("blk.%d.%s", il, n) }
		heads, kvHeads, d := int64(cfg.Heads), int64(cfg.HeadsKV[il]), int64(cfg.HeadDim[il])
		group := heads / kvHeads
		c, sn, m := cos, sin, mask
		if cfg.SWA[il] {
			c, sn, m = cosSWA, sinSWA, maskSWA
		}

		h, err := b.rms(x, blk("attn_norm.weight"), 1)
		if err != nil {
			return err
		}
		q, err := b.linear(h, blk("attn_q.weight"))
		if err != nil {
			return err
		}
		k, err := b.linear(h, blk("attn_k.weight"))
		if err != nil {
			return err
		}
		v, err := b.linear(h, blk("attn_v.weight"))
		if err != nil {
			return err
		}
		if q, err = b.rms(g.Reshape(q, 0, 0, heads, d), blk("attn_q_norm.weight"), 1); err != nil {
			return err
		}
		if k, err = b.rms(g.Reshape(k, 0, 0, kvHeads, d), blk("attn_k_norm.weight"), 1); err != nil {
			return err
		}
		if v, err = b.rms(g.Reshape(v, 0, 0, kvHeads, d), "", 1); err != nil {
			return err
		}
		q = b.rope(q, c, sn, d)
		k = b.rope(k, c, sn, d)

		// Query head h uses key head h / group, as ggml broadcasts them.
		q = g.Transpose(g.Reshape(q, 0, 0, kvHeads, group, d), 0, 2, 3, 1, 4) // [1, KV, G, L, D]
		k = g.Reshape(g.Transpose(k, 0, 2, 1, 3), 0, kvHeads, 1, -1, d)       // [1, KV, 1, L, D]
		v = g.Reshape(g.Transpose(v, 0, 2, 1, 3), 0, kvHeads, 1, -1, d)
		att := b.attend(q, k, v, m, cfg.SWA[il])                        // [1, KV, G, L, D]
		att = g.Reshape(g.Transpose(att, 0, 3, 1, 2, 4), 0, 0, heads*d) // [1, L, H*D]
		if att, err = b.linear(att, blk("attn_output.weight")); err != nil {
			return err
		}
		if att, err = b.rms(att, blk("post_attention_norm.weight"), 1/s); err != nil {
			return err
		}
		x = g.Add(att, x)

		f, err := b.rms(x, blk("ffn_norm.weight"), 1)
		if err != nil {
			return err
		}
		gate, err := b.linear(f, blk("ffn_gate.weight"))
		if err != nil {
			return err
		}
		up, err := b.linear(f, blk("ffn_up.weight"))
		if err != nil {
			return err
		}
		if f, err = b.linear(g.Mul(g.Gelu(gate), up), blk("ffn_down.weight")); err != nil {
			return err
		}
		if f, err = b.rms(f, blk("post_ffw_norm.weight"), 1/s); err != nil {
			return err
		}
		x = g.Add(f, x)

		// The one reader of the residual stream that is not a norm.
		peIn := x
		if s != 1 {
			peIn = g.Mul(x, g.Scalar(s))
		}
		pe, err := b.linear(peIn, blk("inp_gate.weight"))
		if err != nil {
			return err
		}
		pin, err := perLayerInput(il)
		if err != nil {
			return err
		}
		pe = g.Mul(g.Gelu(pe), pin)
		if pe, err = b.linear(pe, blk("proj.weight")); err != nil {
			return err
		}
		if pe, err = b.rms(pe, blk("post_norm.weight"), 1/s); err != nil {
			return err
		}
		x = g.Add(x, pe)

		scale, err := b.vector(blk("layer_output_scale.weight"), 1)
		if err != nil {
			return err
		}
		x = g.Mul(x, scale)
	}
	out, err := b.rms(x, "output_norm.weight", 1)
	if err != nil {
		return err
	}
	// Projecting every token and pooling after is what llama.cpp does; the
	// projection is linear, so it is the same as pooling first.
	if out, err = b.linear(out, "output.weight"); err != nil {
		return err
	}
	g.Result(OutEmbeds, out)
	return nil
}
