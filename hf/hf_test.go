package hf

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestParseSpec(t *testing.T) {
	for in, want := range map[string]Spec{
		"unsloth/embeddinggemma-2-GGUF:Q8_0":       {"unsloth/embeddinggemma-2-GGUF", "Q8_0"},
		"unsloth/embeddinggemma-2-GGUF":            {"unsloth/embeddinggemma-2-GGUF", "Q8_0"},
		"unsloth/embeddinggemma-2-GGUF:UD-Q6_K_XL": {"unsloth/embeddinggemma-2-GGUF", "UD-Q6_K_XL"},
	} {
		got, err := ParseSpec(in)
		if err != nil || got != want {
			t.Errorf("ParseSpec(%q) = %v, %v, want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"", "repo", "/repo", "owner/", "a/b/c", "../x", "a/..", "a/b:../c"} {
		if _, err := ParseSpec(in); err == nil {
			t.Errorf("ParseSpec(%q) did not fail", in)
		}
	}
}

// repo serves a repository listing and its files as Hugging Face does.
type repo struct {
	files     map[string]string
	badSHA    bool
	downloads atomic.Int32
}

func (r *repo) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch {
	case req.URL.Path == "/api/models/owner/m-GGUF/tree/main":
		var tree []map[string]any
		tree = append(tree, map[string]any{"type": "file", "path": "README.md", "size": 3})
		for name, content := range r.files {
			sum := sha256.Sum256([]byte(content))
			oid := hex.EncodeToString(sum[:])
			if r.badSHA {
				oid = strings.Repeat("0", 64)
			}
			tree = append(tree, map[string]any{"type": "file", "path": name, "size": len(content),
				"lfs": map[string]any{"oid": oid, "size": len(content)}})
		}
		json.NewEncoder(w).Encode(tree)
	case strings.HasPrefix(req.URL.Path, "/owner/m-GGUF/resolve/main/"):
		content, ok := r.files[strings.TrimPrefix(req.URL.Path, "/owner/m-GGUF/resolve/main/")]
		if !ok {
			http.NotFound(w, req)
			return
		}
		r.downloads.Add(1)
		w.Write([]byte(content))
	default:
		http.NotFound(w, req)
	}
}

func TestFetch(t *testing.T) {
	r := &repo{files: map[string]string{
		"mmproj-Q8_0.gguf": "projector",
		"m-Q8_0.gguf":      "the model",
		"m-F16.gguf":       "the model, wider",
	}}
	srv := httptest.NewServer(r)
	defer srv.Close()
	f := &Fetcher{Dir: t.TempDir(), Base: srv.URL}
	s := Spec{Repo: "owner/m-GGUF", Quant: "q8_0"}
	got, err := f.Fetch(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(f.Dir, "owner_m-GGUF", "m-Q8_0.gguf"); got != want {
		t.Fatalf("path = %s, want %s", got, want)
	}
	if b, _ := os.ReadFile(got); string(b) != "the model" {
		t.Fatalf("content = %q", b)
	}
	// The second fetch finds the file without the network.
	srv.Close()
	again, err := f.Fetch(context.Background(), s)
	if err != nil || again != got {
		t.Fatalf("second fetch = %s, %v", again, err)
	}
	if n := r.downloads.Load(); n != 1 {
		t.Fatalf("%d downloads, want 1", n)
	}
}

func TestFetchRefuses(t *testing.T) {
	t.Run("missing quantization", func(t *testing.T) {
		srv := httptest.NewServer(&repo{files: map[string]string{"m-F16.gguf": "x", "mmproj-Q4_0.gguf": "y"}})
		defer srv.Close()
		f := &Fetcher{Dir: t.TempDir(), Base: srv.URL}
		_, err := f.Fetch(context.Background(), Spec{Repo: "owner/m-GGUF", Quant: "Q4_0"})
		if err == nil || !strings.Contains(err.Error(), "m-F16.gguf") {
			t.Fatalf("err = %v, want one naming what the repository has", err)
		}
	})
	t.Run("wrong sha256", func(t *testing.T) {
		srv := httptest.NewServer(&repo{files: map[string]string{"m-Q8_0.gguf": "x"}, badSHA: true})
		defer srv.Close()
		f := &Fetcher{Dir: t.TempDir(), Base: srv.URL}
		_, err := f.Fetch(context.Background(), Spec{Repo: "owner/m-GGUF", Quant: "Q8_0"})
		if err == nil || !strings.Contains(err.Error(), "sha256") {
			t.Fatalf("err = %v, want a sha256 mismatch", err)
		}
		entries, _ := os.ReadDir(filepath.Join(f.Dir, "owner_m-GGUF"))
		if len(entries) != 0 {
			t.Fatalf("left %v behind", entries)
		}
	})
}

func TestFetchFiles(t *testing.T) {
	r := &repo{files: map[string]string{
		"int8/a.xml":    "graph",
		"weights/b.npy": "weights",
		"c.json":        "{}",
	}}
	srv := httptest.NewServer(r)
	defer srv.Close()
	f := &Fetcher{Dir: t.TempDir(), Base: srv.URL}
	want := []string{"int8/a.xml", "weights/b.npy"}
	dir, err := f.FetchFiles(context.Background(), "owner/m-GGUF", want)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "weights", "b.npy")); string(b) != "weights" {
		t.Fatalf("content = %q", b)
	}
	if _, err := os.Stat(filepath.Join(dir, "c.json")); err == nil {
		t.Fatal("fetched a file that was not asked for")
	}
	// The second fetch finds the files without the network.
	srv.Close()
	again, err := f.FetchFiles(context.Background(), "owner/m-GGUF", want)
	if err != nil || again != dir {
		t.Fatalf("second fetch = %s, %v", again, err)
	}
	if n := r.downloads.Load(); n != 2 {
		t.Fatalf("%d downloads, want 2", n)
	}
	for _, bad := range []string{"../x", "/etc/passwd", `a\b`} {
		if _, err := f.FetchFiles(context.Background(), "owner/m-GGUF", []string{bad}); err == nil {
			t.Errorf("FetchFiles(%q) did not fail", bad)
		}
	}
}
