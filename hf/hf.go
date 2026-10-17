// Package hf downloads a GGUF from a Hugging Face model repository, given as
// llama-server's -hf takes it: "owner/repo:QUANT", or any set of files from
// one with FetchFiles.
package hf

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

// DefaultBase is where repositories are fetched from.
const DefaultBase = "https://huggingface.co"

// Fetcher downloads model files into Dir, one directory per repository.
type Fetcher struct {
	Dir    string
	Base   string       // DefaultBase when empty
	Client *http.Client // http.DefaultClient when nil
	Log    func(format string, args ...any)
}

// Spec is a repository and the quantization wanted from it.
type Spec struct {
	Repo  string // owner/name
	Quant string // Q8_0
}

// ParseSpec reads "owner/repo:QUANT"; the quantization defaults to Q8_0.
func ParseSpec(s string) (Spec, error) {
	repo, quant, _ := strings.Cut(s, ":")
	if quant == "" {
		quant = "Q8_0"
	}
	owner, name, ok := strings.Cut(repo, "/")
	if !ok || owner == "" || name == "" || strings.ContainsAny(name, "/\\") || strings.HasPrefix(owner, ".") ||
		strings.HasPrefix(name, ".") {
		return Spec{}, fmt.Errorf("hf: %q is not owner/repo[:QUANT]", s)
	}
	if strings.ContainsAny(quant, "/\\") {
		return Spec{}, fmt.Errorf("hf: %q is not a quantization", quant)
	}
	return Spec{Repo: repo, Quant: quant}, nil
}

func (s Spec) String() string { return s.Repo + ":" + s.Quant }

// matches says whether a file in the repository is the model at this
// quantization: "<anything>-Q8_0.gguf", and not a multimodal projector, whose
// name ends the same way.
func (s Spec) matches(name string) bool {
	name = strings.ToLower(path.Base(name))
	return strings.HasSuffix(name, "-"+strings.ToLower(s.Quant)+".gguf") && !strings.HasPrefix(name, "mmproj")
}

func (f *Fetcher) dir(s Spec) string {
	return filepath.Join(f.Dir, strings.ReplaceAll(s.Repo, "/", "_"))
}

// Fetch returns the local path of the model file, downloading it first if it
// is not already there. A file already downloaded is used without asking the
// network, so a start does not depend on Hugging Face being up.
func (f *Fetcher) Fetch(ctx context.Context, s Spec) (string, error) {
	dir := f.dir(s)
	if entries, err := os.ReadDir(dir); err == nil {
		for _, e := range entries {
			if e.Type().IsRegular() && s.matches(e.Name()) {
				return filepath.Join(dir, e.Name()), nil
			}
		}
	}
	file, err := f.find(ctx, s)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	dst := filepath.Join(dir, path.Base(file.Path))
	if err := f.download(ctx, s.Repo, file, dst); err != nil {
		return "", err
	}
	return dst, nil
}

// FetchFiles returns the local directory holding the given files of a
// repository, laid out as they are in it ("int8/model.xml"), downloading the
// ones not already there. When every file is present the network is not
// asked, as with Fetch.
func (f *Fetcher) FetchFiles(ctx context.Context, repo string, files []string) (string, error) {
	if _, err := ParseSpec(repo); err != nil || strings.Contains(repo, ":") {
		return "", fmt.Errorf("hf: %q is not owner/repo", repo)
	}
	dir := filepath.Join(f.Dir, strings.ReplaceAll(repo, "/", "_"))
	var missing []string
	for _, name := range files {
		if !filepath.IsLocal(name) || strings.Contains(name, "\\") {
			return "", fmt.Errorf("hf: %q is not a path in a repository", name)
		}
		if st, err := os.Stat(filepath.Join(dir, name)); err != nil || !st.Mode().IsRegular() {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		return dir, nil
	}
	res, err := f.get(ctx, f.base()+"/api/models/"+repo+"/tree/main?recursive=1")
	if err != nil {
		return "", err
	}
	var tree []treeEntry
	err = json.NewDecoder(res.Body).Decode(&tree)
	res.Body.Close()
	if err != nil {
		return "", fmt.Errorf("hf: listing %s: %w", repo, err)
	}
	byPath := make(map[string]treeEntry, len(tree))
	for _, e := range tree {
		if e.Type == "file" {
			byPath[e.Path] = e
		}
	}
	for _, name := range missing {
		e, ok := byPath[filepath.ToSlash(name)]
		if !ok {
			return "", fmt.Errorf("hf: %s has no %s", repo, name)
		}
		dst := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return "", err
		}
		if err := f.download(ctx, repo, e, dst); err != nil {
			return "", err
		}
	}
	return dir, nil
}

type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID  string `json:"oid"` // sha256 of the content
		Size int64  `json:"size"`
	} `json:"lfs"`
}

func (f *Fetcher) base() string {
	if f.Base != "" {
		return strings.TrimRight(f.Base, "/")
	}
	return DefaultBase
}

func (f *Fetcher) client() *http.Client {
	if f.Client != nil {
		return f.Client
	}
	return http.DefaultClient
}

func (f *Fetcher) log(format string, args ...any) {
	if f.Log != nil {
		f.Log(format, args...)
	}
}

func (f *Fetcher) get(ctx context.Context, u string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	res, err := f.client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("hf: %w", err)
	}
	if res.StatusCode != http.StatusOK {
		res.Body.Close()
		return nil, fmt.Errorf("hf: GET %s: %s", u, res.Status)
	}
	return res, nil
}

// find lists the repository's top level and picks the file for the
// quantization.
func (f *Fetcher) find(ctx context.Context, s Spec) (treeEntry, error) {
	res, err := f.get(ctx, f.base()+"/api/models/"+s.Repo+"/tree/main")
	if err != nil {
		return treeEntry{}, err
	}
	defer res.Body.Close()
	var tree []treeEntry
	if err := json.NewDecoder(res.Body).Decode(&tree); err != nil {
		return treeEntry{}, fmt.Errorf("hf: listing %s: %w", s.Repo, err)
	}
	var ggufs []string
	for _, e := range tree {
		if e.Type != "file" || !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		if s.matches(e.Path) {
			return e, nil
		}
		ggufs = append(ggufs, e.Path)
	}
	return treeEntry{}, fmt.Errorf("hf: %s has no %s GGUF (it has %s)", s.Repo, s.Quant, strings.Join(ggufs, ", "))
}

// download writes the file next to dst and renames it into place once its
// size and sha256 are what the repository says, so an interrupted download
// is never taken for the model.
func (f *Fetcher) download(ctx context.Context, repo string, e treeEntry, dst string) error {
	u := f.base() + "/" + repo + "/resolve/main/" + (&url.URL{Path: e.Path}).EscapedPath()
	size, want := e.Size, ""
	if e.LFS != nil {
		size, want = e.LFS.Size, e.LFS.OID
	}
	f.log("downloading %s (%d MB) to %s", u, size>>20, dst)
	started := time.Now()
	res, err := f.get(ctx, u)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), res.Body)
	if err = errors.Join(err, tmp.Close()); err != nil {
		return fmt.Errorf("hf: downloading %s: %w", e.Path, err)
	}
	if n != size {
		return fmt.Errorf("hf: %s is %d bytes, want %d", e.Path, n, size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); want != "" && got != want {
		return fmt.Errorf("hf: %s has sha256 %s, want %s", e.Path, got, want)
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), dst); err != nil {
		return err
	}
	f.log("downloaded %s in %v", e.Path, time.Since(started).Round(time.Second))
	return nil
}
