// Package testmodel finds the GGUF and the llama.cpp fixture the tests use.
package testmodel

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/daaku/eg2npu/gguf"
)

// Path is the EmbeddingGemma 2 Q8_0 GGUF: $EG2_GGUF, or where the readme
// says to download it.
func Path() string {
	if p := os.Getenv("EG2_GGUF"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "eg2", "embeddinggemma-2-Q8_0.gguf")
}

// Open opens the GGUF, or skips the test saying what it needed.
func Open(t testing.TB) *gguf.File {
	t.Helper()
	if _, err := os.Stat(Path()); err != nil {
		t.Skipf("needs the EmbeddingGemma 2 Q8_0 GGUF at %s (or $EG2_GGUF)", Path())
	}
	g, err := gguf.Open(Path())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

// Fixture is what llama.cpp made of the inputs in testdata/llama.json:
// llama-tokenize's ids, and llama-server's vectors (--pooling mean).
type Fixture struct {
	Source   string `json:"source"`
	Tokenize []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	} `json:"tokenize"`
	Embed []struct {
		Text   string    `json:"text"`
		Vector []float32 `json:"vector"`
	} `json:"embed"`
}

// Load reads testdata/llama.json.
func Load(t testing.TB) Fixture {
	t.Helper()
	// From the source tree, or beside a test binary run in a directory of its
	// own (on a machine with an NPU, say).
	_, file, _, _ := runtime.Caller(0)
	data, err := os.ReadFile(filepath.Join(filepath.Dir(file), "..", "..", "testdata", "llama.json"))
	if err != nil {
		data, err = os.ReadFile(filepath.Join("..", "testdata", "llama.json"))
	}
	if err != nil {
		t.Fatal(err)
	}
	var f Fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}
