package eg2

import (
	"math"
	"os"
	"testing"

	"github.com/daaku/eg2npu/internal/testmodel"
)

// The angle is carried in float32 and stepped, as ggml does, so position 0
// is no rotation and the first pair turns by a whole radian a position.
func TestRopeTable(t *testing.T) {
	cos, sin := ropeTable(3, 8, 10000)
	if cos[0] != 1 || sin[3] != 0 {
		t.Errorf("position 0 rotates: %v %v", cos[:4], sin[:4])
	}
	if got := sin[4]; math.Abs(float64(got)-math.Sin(1)) > 1e-7 {
		t.Errorf("position 1, pair 0: %g", got)
	}
	want := math.Cos(2 * math.Pow(10000, -2.0/8*3))
	if got := cos[2*4+3]; math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("position 2, pair 3: %g, want %g", got, want)
	}
}

func TestPool(t *testing.T) {
	v := pool([]float32{1, 2, 5, 6}, 2, 2) // mean [3, 4]
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Errorf("pool = %v", v)
	}
}

func cosine(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / math.Sqrt(na*nb)
}

func open(t *testing.T, opt Options) *Model {
	t.Helper()
	testmodel.Open(t) // skips without the GGUF
	if opt.Device == "" {
		opt.Device = "CPU"
	}
	if dev := os.Getenv("EG2_DEVICE"); dev != "" {
		opt.Device = dev
	}
	opt.CacheDir = t.TempDir()
	m, err := Open(testmodel.Path(), opt)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	return m
}

// The vectors are llama-server's to within its own quantization noise. Q8_0
// keeps llama.cpp's int8 weights; f16 rounds them, and both land as close.
// EG2_DEVICE=NPU runs this on the NPU, with static lengths.
func TestMatchesLlama(t *testing.T) {
	for _, w := range []Weights{WeightsF16, WeightsQ8} {
		t.Run(string(w), func(t *testing.T) {
			opt := Options{Weights: w, Dynamic: true, MaxTokens: 2048}
			if os.Getenv("EG2_DEVICE") == "NPU" {
				opt.Dynamic = false
			}
			m := open(t, opt)
			if err := m.SelfCheck(); err != nil {
				t.Fatal(err)
			}
			for i, c := range testmodel.Load(t).Embed {
				r, err := m.Embed(c.Text)
				if err != nil {
					t.Fatal(err)
				}
				if got := cosine(r.Vector, c.Vector); got < 0.9998 {
					t.Errorf("input %d (%d tokens): cosine %.6f to llama-server's", i, r.Tokens, got)
				}
			}
		})
	}
}

// Attention in blocks of query rows, with the sliding window layers reading
// only the keys in reach, is the same attention: the fixture's input past
// the window, at a static length, attended whole and in blocks of 64 rows.
func TestBlockedAttention(t *testing.T) {
	text := ""
	for _, c := range testmodel.Load(t).Embed {
		if len(c.Text) > len(text) {
			text = c.Text
		}
	}
	var vectors [][]float32
	for _, scores := range []int64{1 << 30, 64 * 64} {
		m := open(t, Options{Lengths: []int{640}, MaxScores: scores, Weights: WeightsQ8})
		r, err := m.Embed(text)
		if err != nil {
			t.Fatal(err)
		}
		if !m.Config.NeedsWindow(r.Tokens) {
			t.Fatalf("%d tokens do not reach past the window", r.Tokens)
		}
		vectors = append(vectors, r.Vector)
	}
	if got := cosine(vectors[0], vectors[1]); got < 0.99999 {
		t.Errorf("blocked attention at cosine %.7f to whole", got)
	}
}

func TestTooLong(t *testing.T) {
	m := open(t, Options{Lengths: []int{32}, MaxTokens: 2048})
	if _, err := m.EmbedTokens(make([]int32, 33)); err == nil {
		t.Error("33 tokens went into a 32 token graph")
	}
}

// Carrying the residual stream divided by a scale changes nothing a norm can
// see, which is every reader of it but the per layer gate, which scales back.
func TestResidualScale(t *testing.T) {
	m := open(t, Options{Dynamic: true, MaxTokens: 2048, Weights: WeightsQ8, ResidualScale: 64})
	for i, c := range testmodel.Load(t).Embed[:3] {
		r, err := m.Embed(c.Text)
		if err != nil {
			t.Fatal(err)
		}
		if got := cosine(r.Vector, c.Vector); got < 0.9998 {
			t.Errorf("input %d: cosine %.6f to llama-server's", i, got)
		}
	}
}
