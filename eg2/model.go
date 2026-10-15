package eg2

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/daaku/eg2npu/gguf"
	"github.com/daaku/eg2npu/ir"
	"github.com/daaku/eg2npu/tokenizer"
	"github.com/daaku/openvino"
)

// graphVersion is part of the IR cache key: bump it whenever Build changes
// what it writes.
const graphVersion = 3

// Options configure a Model.
type Options struct {
	// Device is the OpenVINO device: NPU, GPU or CPU.
	Device string
	// CacheDir holds the IR written from the GGUF and the devices' compiled
	// blobs. It is created when missing.
	CacheDir string
	// Lengths are the sequence lengths compiled for a device that needs static
	// shapes, shortest first; an input is padded to the first that holds it.
	// The last is the longest input the model takes.
	Lengths []int
	// Dynamic compiles one graph for every length instead, which the CPU and
	// GPU can do and the NPU cannot.
	Dynamic bool
	// MaxTokens bounds the input, in tokens with BOS and EOS. 0 is the longest
	// static length.
	MaxTokens int
	Weights   Weights
	// MaxScores: see BuildOptions. 0 means DefaultMaxScores.
	MaxScores int64
	// ResidualScale: see BuildOptions.
	ResidualScale float32
	// Props are extra OpenVINO properties for the compile.
	Props map[string]string
	// Log, when set, hears what the model is doing (compiles, mostly).
	Log func(format string, args ...any)
}

// DefaultMaxScores keeps a block of attention to 768 by 768 scores, the
// largest whole attention the NPU ran.
const DefaultMaxScores = 768 * 768

// DefaultLengths are the static lengths: queries are short, and padding a
// 20 token query to 2048 would cost a hundred times the work.
//
// They avoid the model's hidden size (512) and feed forward width (2048): on
// Panther Lake's NPU (driver 1.38, OpenVINO 2026.4) a graph whose sequence
// length was either one compiled, then faulted the device at its first
// inference (zeFenceHostSynchronize: ZE_RESULT_ERROR_UNKNOWN), while 504,
// 576, 1024, 1040 and 2064 ran. 2064 holds llama-server's 2048 token limit.
var DefaultLengths = []int{32, 64, 128, 256, 384, 640, 1024, 1536, 2064}

type compiled struct {
	mu     sync.Mutex
	length int // -1 when dynamic
	model  *openvino.CompiledModel
	req    *openvino.Request
}

// Model is EmbeddingGemma 2's text encoder on one device.
type Model struct {
	Config    Config
	Tokenizer *tokenizer.Tokenizer
	opt       Options
	file      *gguf.File
	embd      *gguf.Tensor
	core      *openvino.Core
	dir       string // the IR

	mu       sync.Mutex
	compiled map[int]*compiled
	rope     sync.Map // length -> *ropeTables
}

// Open reads the GGUF and writes its IR into the cache if it is not there.
// Graphs are compiled when first used, or by Warm.
func Open(path string, opt Options) (*Model, error) {
	if opt.Device == "" {
		opt.Device = "CPU"
	}
	if len(opt.Lengths) == 0 {
		opt.Lengths = DefaultLengths
	}
	if opt.MaxTokens == 0 {
		opt.MaxTokens = opt.Lengths[len(opt.Lengths)-1]
	}
	if opt.Weights == "" {
		opt.Weights = WeightsF16
	}
	if opt.ResidualScale == 0 {
		opt.ResidualScale = 1
	}
	if opt.MaxScores == 0 {
		opt.MaxScores = DefaultMaxScores
	}
	if opt.Log == nil {
		opt.Log = func(string, ...any) {}
	}
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	m := &Model{opt: opt, file: f, compiled: map[int]*compiled{}}
	if m.Config, err = ConfigFromGGUF(f); err != nil {
		f.Close()
		return nil, err
	}
	if m.Tokenizer, err = tokenizer.FromGGUF(f); err != nil {
		f.Close()
		return nil, err
	}
	if m.embd, err = f.Tensor("token_embd.weight"); err != nil {
		f.Close()
		return nil, err
	}
	if m.core, err = openvino.SharedCore(); err != nil {
		f.Close()
		return nil, err
	}
	if err := m.writeIR(path); err != nil {
		f.Close()
		return nil, err
	}
	return m, nil
}

// Close releases the compiled models and the GGUF.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range m.compiled {
		c.req.Close()
		c.model.Close()
	}
	m.compiled = nil
	m.file.Close()
}

// MaxTokens is the longest input, in tokens with BOS and EOS, the model takes.
func (m *Model) MaxTokens() int {
	if m.opt.Dynamic {
		return m.opt.MaxTokens
	}
	return min(m.opt.MaxTokens, m.opt.Lengths[len(m.opt.Lengths)-1])
}

// irKey names the IR in the cache: the GGUF's identity and everything that
// changes what Build writes.
func (m *Model) irKey(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	// The header and the first tensors identify the file without reading all
	// of it.
	if _, err := io.CopyN(h, f, min(st.Size(), 16<<20)); err != nil {
		return "", err
	}
	fmt.Fprintf(h, "|%d|v%d|%s|%g|%d", st.Size(), graphVersion, m.opt.Weights, m.opt.ResidualScale,
		m.opt.MaxScores)
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

func (m *Model) xmlName(length int) string {
	if length < 0 {
		return "dynamic.xml"
	}
	return fmt.Sprintf("L%d.xml", length)
}

func (m *Model) writeIR(path string) error {
	key, err := m.irKey(path)
	if err != nil {
		return err
	}
	m.dir = filepath.Join(m.opt.CacheDir, "ir-"+key)
	want := []int{-1}
	want = append(want, m.opt.Lengths...)
	missing := false
	for _, l := range want {
		if _, err := os.Stat(filepath.Join(m.dir, m.xmlName(l))); err != nil {
			missing = true
		}
	}
	if _, err := os.Stat(filepath.Join(m.dir, "weights.bin")); err != nil {
		missing = true
	}
	if !missing {
		return nil
	}
	// Every graph is written again, so the one weights file holds the
	// constants of all of them.
	started := time.Now()
	tmp, err := os.MkdirTemp(m.opt.CacheDir, "ir-tmp-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	pool := ir.NewPool()
	for _, l := range want {
		g, err := Build(m.file, m.Config, pool, BuildOptions{
			Length: int64(l), Weights: m.opt.Weights, ResidualScale: m.opt.ResidualScale,
			MaxScores: m.opt.MaxScores,
		})
		if err != nil {
			return err
		}
		out, err := os.Create(filepath.Join(tmp, m.xmlName(l)))
		if err != nil {
			return err
		}
		if err := errors.Join(g.Write(out), out.Close()); err != nil {
			return err
		}
	}
	if err := os.WriteFile(filepath.Join(tmp, "weights.bin"), pool.Bytes(), 0o644); err != nil {
		return err
	}
	os.RemoveAll(m.dir)
	if err := os.Rename(tmp, m.dir); err != nil {
		return err
	}
	m.opt.Log("wrote the IR to %s in %v", m.dir, time.Since(started).Round(time.Millisecond))
	return nil
}

// length picks the compiled length for n tokens.
func (m *Model) length(n int) (int, error) {
	if n > m.MaxTokens() {
		return 0, fmt.Errorf("input (%d tokens) is too large to process. The longest input is %d tokens",
			n, m.MaxTokens())
	}
	if m.opt.Dynamic {
		return -1, nil
	}
	for _, l := range m.opt.Lengths {
		if l >= n {
			return l, nil
		}
	}
	panic("unreachable")
}

func (m *Model) get(length int) (*compiled, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if c, ok := m.compiled[length]; ok {
		return c, nil
	}
	started := time.Now()
	model, err := m.core.ReadModel(filepath.Join(m.dir, m.xmlName(length)), filepath.Join(m.dir, "weights.bin"))
	if err != nil {
		return nil, err
	}
	defer model.Close()
	props := map[string]string{"CACHE_DIR": filepath.Join(m.opt.CacheDir, "blobs")}
	if m.opt.Device == "CPU" {
		// A CPU with AMX or AVX512-BF16 would otherwise run in bfloat16.
		props["INFERENCE_PRECISION_HINT"] = "f32"
	}
	for k, v := range m.opt.Props {
		props[k] = v
	}
	cm, err := m.core.CompileModel(model, m.opt.Device, props)
	if err != nil {
		return nil, fmt.Errorf("eg2: compiling length %d for %s: %w", length, m.opt.Device, err)
	}
	req, err := cm.Request()
	if err != nil {
		cm.Close()
		return nil, err
	}
	c := &compiled{length: length, model: cm, req: req}
	m.compiled[length] = c
	m.opt.Log("compiled %s for %s in %v", m.xmlName(length), m.opt.Device, time.Since(started).Round(time.Millisecond))
	return c, nil
}

// SelfCheck embeds one text at every compiled length and holds the results to
// each other, so a graph the device miscompiles or faults on is found at
// startup rather than by a query. The vectors differ only by padding, which
// the mask hides.
func (m *Model) SelfCheck() error {
	ids := m.Tokenizer.Encode("task: search result | query: does every compiled length agree?")
	lengths := []int{-1}
	if !m.opt.Dynamic {
		lengths = m.opt.Lengths
	}
	var first []float32
	for _, l := range lengths {
		r, err := m.embedAt(ids, l)
		if err != nil {
			return fmt.Errorf("eg2: self check at length %d: %w", l, err)
		}
		var norm float64
		for _, v := range r.Vector {
			norm += float64(v) * float64(v)
		}
		if math.IsNaN(norm) || math.Abs(norm-1) > 1e-3 {
			return fmt.Errorf("eg2: self check at length %d: vector of norm %g", l, math.Sqrt(norm))
		}
		if first == nil {
			first = r.Vector
			continue
		}
		var dot float64
		for i := range first {
			dot += float64(first[i]) * float64(r.Vector[i])
		}
		if dot < 0.999 {
			return fmt.Errorf("eg2: self check: length %d gives a vector at cosine %.4f to length %d's",
				l, dot, lengths[0])
		}
	}
	return nil
}

// Warm compiles every graph up front, so the first queries do not wait.
func (m *Model) Warm() error {
	if m.opt.Dynamic {
		_, err := m.get(-1)
		return err
	}
	for _, l := range m.opt.Lengths {
		if _, err := m.get(l); err != nil {
			return err
		}
	}
	return nil
}

type ropeTables struct {
	cos, sin, cosSWA, sinSWA []float32
}

// ropeTable is ggml's rope cache for n positions of a head of d: the angle is
// carried in float32 and multiplied by base^(-2/d) per pair, as
// ggml_rope_cache_init does, so the angles round as llama.cpp's do.
func ropeTable(n, d int, base float64) (cos, sin []float32) {
	half := d / 2
	cos = make([]float32, n*half)
	sin = make([]float32, n*half)
	scale := float32(math.Pow(base, -2.0/float64(d)))
	for p := range n {
		theta := float32(p)
		for i := range half {
			cos[p*half+i] = float32(math.Cos(float64(theta)))
			sin[p*half+i] = float32(math.Sin(float64(theta)))
			theta *= scale
		}
	}
	return cos, sin
}

func (m *Model) ropeFor(n int) *ropeTables {
	if t, ok := m.rope.Load(n); ok {
		return t.(*ropeTables)
	}
	var dimFull, dimSWA int
	for i := range m.Config.Layers {
		if m.Config.SWA[i] {
			dimSWA = m.Config.HeadDim[i]
		} else {
			dimFull = m.Config.HeadDim[i]
		}
	}
	t := &ropeTables{}
	t.cos, t.sin = ropeTable(n, dimFull, m.Config.RopeBase)
	t.cosSWA, t.sinSWA = ropeTable(n, dimSWA, m.Config.RopeBaseSWA)
	m.rope.Store(n, t)
	return t
}

// Result is one input's embedding.
type Result struct {
	Vector []float32
	Tokens int
	Length int // the compiled length it ran at, -1 when dynamic
}

// Embed embeds one text, which should carry its task prompt already.
func (m *Model) Embed(text string) (Result, error) {
	return m.EmbedTokens(m.Tokenizer.Encode(text))
}

// EmbedTokens embeds token ids: mean pooled over the tokens and L2
// normalized, as llama-server with --pooling mean answers.
func (m *Model) EmbedTokens(ids []int32) (Result, error) {
	n := len(ids)
	if n == 0 {
		return Result{}, errors.New("eg2: no tokens")
	}
	length, err := m.length(n)
	if err != nil {
		return Result{}, err
	}
	return m.embedAt(ids, length)
}

// embedAt embeds at a given compiled length.
func (m *Model) embedAt(ids []int32, length int) (Result, error) {
	n := len(ids)
	c, err := m.get(length)
	if err != nil {
		return Result{}, err
	}
	L := length
	if L < 0 {
		L = n
	}
	cfg := m.Config
	embd := cfg.Embd

	c.mu.Lock()
	defer c.mu.Unlock()
	var tensors []*openvino.Tensor
	defer func() {
		for _, t := range tensors {
			t.Close()
		}
	}()
	input := func(name string, shape []int64) ([]float32, error) {
		if !m.hasInput(c, name) {
			return nil, nil
		}
		t, data, err := openvino.NewF32Tensor(shape)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, t)
		if err := c.req.Set(name, t); err != nil {
			return nil, err
		}
		return data, nil
	}

	embeds, err := input(InEmbeds, []int64{1, int64(L), int64(embd)})
	if err != nil {
		return Result{}, err
	}
	scale := float32(math.Sqrt(float64(embd)))
	for i, id := range ids {
		row := embeds[i*embd : (i+1)*embd]
		m.file.Row(m.embd, int(id), row)
		for j := range row {
			row[j] *= scale
		}
	}
	mask, err := input(InMask, []int64{1, 1, 1, 1, int64(L)})
	if err != nil {
		return Result{}, err
	}
	for j := n; j < L; j++ {
		mask[j] = maskValue
	}
	maskSWA, err := input(InMaskSWA, []int64{1, 1, 1, int64(L), int64(L)})
	if err != nil {
		return Result{}, err
	}
	if maskSWA != nil {
		half := cfg.Window / 2
		for i := range L {
			for j := range L {
				if j >= n || j-i > half || i-j > half {
					maskSWA[i*L+j] = maskValue
				}
			}
		}
	}
	rope := m.ropeFor(L)
	for _, in := range []struct {
		name   string
		values []float32
	}{{InCos, rope.cos}, {InSin, rope.sin}, {InCosSWA, rope.cosSWA}, {InSinSWA, rope.sinSWA}} {
		data, err := input(in.name, []int64{int64(L), int64(len(in.values) / L)})
		if err != nil {
			return Result{}, err
		}
		copy(data, in.values)
	}
	if err := c.req.Infer(); err != nil {
		return Result{}, err
	}
	out, err := c.req.Get(OutEmbeds)
	if err != nil {
		return Result{}, err
	}
	defer out.Close()
	values, err := out.F32()
	if err != nil {
		return Result{}, err
	}
	dim := cfg.EmbdOut
	if len(values) < n*dim {
		return Result{}, fmt.Errorf("eg2: output of %d values for %d tokens", len(values), n)
	}
	return Result{Vector: pool(values[:n*dim], n, dim), Tokens: n, Length: length}, nil
}

// pool is the mean over tokens, L2 normalized as common_embd_normalize does.
func pool(values []float32, n, dim int) []float32 {
	sum := make([]float64, dim)
	for i := range n {
		for j, v := range values[i*dim : (i+1)*dim] {
			sum[j] += float64(v)
		}
	}
	out := make([]float32, dim)
	var norm float64
	for j := range dim {
		out[j] = float32(sum[j] / float64(n))
		norm += float64(out[j]) * float64(out[j])
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for j := range out {
			out[j] *= inv
		}
	}
	return out
}

func (m *Model) hasInput(c *compiled, name string) bool {
	_, err := c.model.Input(name)
	return err == nil
}

// Device is the device the model runs on.
func (m *Model) Device() string { return m.opt.Device }

// Describe says how the model is set up, for logs.
func (m *Model) Describe() string {
	shapes := "lengths " + strings.Trim(fmt.Sprint(m.opt.Lengths), "[]")
	if m.opt.Dynamic {
		shapes = fmt.Sprintf("dynamic up to %d tokens", m.opt.MaxTokens)
	}
	return fmt.Sprintf("%s, %s weights, %s", m.opt.Device, m.opt.Weights, shapes)
}
