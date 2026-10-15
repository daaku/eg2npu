// Package eg2 runs EmbeddingGemma 2's text encoder through OpenVINO, from the
// GGUF file llama-server loads. The graph is llama.cpp's
// (src/models/gemma-embedding2.cpp) written out as OpenVINO IR, so the vectors
// land where llama-server's do.
package eg2

import (
	"fmt"

	"github.com/daaku/eg2npu/gguf"
)

// Config is the shape of the model, read from the GGUF metadata.
type Config struct {
	Layers      int
	Embd        int // hidden size
	EmbdOut     int // the vector's size
	PerLayer    int // per layer input size
	FF          int
	Heads       int
	HeadsKV     []int // per layer
	HeadDim     []int // per layer
	SWA         []bool
	Window      int // llama.cpp's n_swa; a token sees those within Window/2 on either side
	RopeBase    float64
	RopeBaseSWA float64
	Eps         float32
}

// ConfigFromGGUF reads the config of a gemma-embedding2 GGUF.
func ConfigFromGGUF(g *gguf.File) (Config, error) {
	const p = "gemma-embedding2."
	if arch, _ := g.KV["general.architecture"].(string); arch != "gemma-embedding2" {
		return Config{}, fmt.Errorf("eg2: architecture %q, want gemma-embedding2", arch)
	}
	var c Config
	var ok [10]bool
	c.Layers, ok[0] = g.Uint(p + "block_count")
	c.Embd, ok[1] = g.Uint(p + "embedding_length")
	c.EmbdOut, ok[2] = g.Uint(p + "embedding_length_out")
	c.PerLayer, ok[3] = g.Uint(p + "embedding_length_per_layer_input")
	c.FF, ok[4] = g.Uint(p + "feed_forward_length")
	c.Heads, ok[5] = g.Uint(p + "attention.head_count")
	c.Window, ok[6] = g.Uint(p + "attention.sliding_window")
	c.RopeBase, ok[7] = g.Float(p + "rope.freq_base")
	c.RopeBaseSWA, ok[8] = g.Float(p + "rope.freq_base_swa")
	eps, ok9 := g.Float(p + "attention.layer_norm_rms_epsilon")
	ok[9] = ok9
	c.Eps = float32(eps)
	for i, v := range ok {
		if !v {
			return Config{}, fmt.Errorf("eg2: GGUF metadata %d missing", i)
		}
	}
	keyLen, ok1 := g.Uint(p + "attention.key_length")
	keyLenSWA, ok2 := g.Uint(p + "attention.key_length_swa")
	ropeDim, ok3 := g.Uint(p + "rope.dimension_count")
	ropeDimSWA, ok4 := g.Uint(p + "rope.dimension_count_swa")
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return Config{}, fmt.Errorf("eg2: GGUF lacks head or rope sizes")
	}
	// The graph rotates the whole head; a partial rotation would need another
	// graph.
	if ropeDim != keyLen || ropeDimSWA != keyLenSWA {
		return Config{}, fmt.Errorf("eg2: partial rotary embeddings (%d of %d) are not supported", ropeDim, keyLen)
	}
	pattern, _ := g.KV[p+"attention.sliding_window_pattern"].([]any)
	kv, _ := g.KV[p+"attention.head_count_kv"].([]any)
	if len(pattern) != c.Layers || len(kv) != c.Layers {
		return Config{}, fmt.Errorf("eg2: GGUF lacks the per layer attention pattern")
	}
	for i := range c.Layers {
		swa, _ := pattern[i].(bool)
		n, _ := gguf.Int(kv[i])
		c.SWA = append(c.SWA, swa)
		c.HeadsKV = append(c.HeadsKV, int(n))
		if swa {
			c.HeadDim = append(c.HeadDim, keyLenSWA)
		} else {
			c.HeadDim = append(c.HeadDim, keyLen)
		}
		if n == 0 || c.Heads%int(n) != 0 {
			return Config{}, fmt.Errorf("eg2: layer %d has %d kv heads for %d heads", i, n, c.Heads)
		}
	}
	return c, nil
}

// NeedsWindow says whether a sequence of n tokens is long enough for the
// sliding window to hide one token from another.
func (c Config) NeedsWindow(n int) bool { return n > c.Window/2+1 }
