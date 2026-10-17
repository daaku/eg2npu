// Command eg2npu serves EmbeddingGemma 2 text embeddings over llama-server's
// API, on an Intel NPU (or the CPU or GPU) through OpenVINO, from the same
// GGUF file llama-server loads.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/daaku/eg2npu/eg2"
	"github.com/daaku/eg2npu/hf"
	"github.com/daaku/eg2npu/server"
	"github.com/daaku/lands"
)

// model adapts eg2.Model to the server.
type model struct{ m *eg2.Model }

func (a model) Embed(text string) ([]float32, int, error) {
	r, err := a.m.Embed(text)
	return r.Vector, r.Tokens, err
}

func (a model) EmbedTokens(ids []int32) ([]float32, int, error) {
	for _, id := range ids {
		if id < 0 || int(id) >= a.m.Tokenizer.Size() {
			return nil, 0, fmt.Errorf("token id %d is out of the vocabulary", id)
		}
	}
	r, err := a.m.EmbedTokens(ids)
	return r.Vector, r.Tokens, err
}

func (a model) MaxTokens() int { return a.m.MaxTokens() }

func cacheDir() string {
	if dir := os.Getenv("CACHE_DIRECTORY"); dir != "" { // systemd's CacheDirectory=
		return dir
	}
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "eg2npu")
}

// modelsDir is where -hf downloads go: systemd's StateDirectory=, so clearing
// the cache does not mean downloading the model again, or the cache directory.
func modelsDir() string {
	if dir := os.Getenv("STATE_DIRECTORY"); dir != "" {
		return filepath.Join(dir, "models")
	}
	return filepath.Join(cacheDir(), "models")
}

func main() {
	modelPath := flag.String("model", "", "the EmbeddingGemma 2 GGUF file (the one llama-server loads); overrides -hf")
	hfSpec := flag.String("hf", "unsloth/embeddinggemma-2-GGUF:Q8_0",
		"Hugging Face repository and quantization to download the GGUF from, as llama-server's -hf")
	models := flag.String("models", modelsDir(), "where -hf downloads are kept")
	device := flag.String("device", "NPU", "OpenVINO device: NPU, GPU or CPU")
	listen := flag.String("listen", "127.0.0.1:8741", "where to serve, unless systemd passes a socket (LISTEN_FDS) or PORT or ADDR is set")
	alias := flag.String("alias", "embeddinggemma2", "the model's name in /v1/models: keep llama-server's --alias")
	cache := flag.String("cache", cacheDir(), "where the IR and the compiled blobs are kept")
	shapes := flag.String("shapes", "auto", "static (one graph per length, needed by the NPU), dynamic, or auto (dynamic on the CPU only)")
	lengths := flag.String("lengths", "", "static lengths, comma separated (default "+
		strings.Trim(strings.ReplaceAll(fmt.Sprint(eg2.DefaultLengths), " ", ","), "[]")+")")
	maxTokens := flag.Int("max-tokens", 2048, "the longest input, in tokens (llama-server's limit is its -ub)")
	weights := flag.String("weights", "auto", "how the IR keeps weights: f16, q8 (Q8_0's own int8 and scales), "+
		"or auto: q8 on the CPU, where it ran a third faster, and f16 elsewhere: the NPU computes in float16 "+
		"either way and compiles q8 2.5x slower, and the GPU ran q8 slower and further from llama-server")
	verbose := flag.Bool("v", false, "log every request")
	flag.Parse()
	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *modelPath == "" {
		spec, err := hf.ParseSpec(*hfSpec)
		if err != nil {
			log.Error("bad -hf", "error", err)
			os.Exit(2)
		}
		f := &hf.Fetcher{Dir: *models, Log: func(format string, args ...any) { log.Info(fmt.Sprintf(format, args...)) }}
		if *modelPath, err = f.Fetch(ctx, spec); err != nil {
			log.Error("fetching the model", "hf", spec, "error", err)
			os.Exit(1)
		}
	}
	opt := eg2.Options{
		Device: strings.ToUpper(*device), CacheDir: *cache, MaxTokens: *maxTokens, Weights: eg2.Weights(*weights),
		Log: func(format string, args ...any) { log.Info(fmt.Sprintf(format, args...)) },
	}
	if *weights == "auto" {
		opt.Weights = eg2.WeightsF16
		if opt.Device == "CPU" {
			opt.Weights = eg2.WeightsQ8
		}
	}
	switch *shapes {
	case "static":
	case "dynamic":
		opt.Dynamic = true
	case "auto":
		// Only the CPU: the NPU cannot run dynamic shapes, and the GPU took 27 ms
		// a query with them when the length changed from one query to the next,
		// against 4.6 ms with static lengths.
		opt.Dynamic = opt.Device == "CPU"
	default:
		log.Error("-shapes is static, dynamic or auto")
		os.Exit(2)
	}
	if *lengths != "" {
		for _, s := range strings.Split(*lengths, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil || n <= 0 {
				log.Error("bad -lengths", "value", s)
				os.Exit(2)
			}
			opt.Lengths = append(opt.Lengths, n)
		}
	}
	if err := os.MkdirAll(*cache, 0o755); err != nil {
		log.Error("cache directory", "error", err)
		os.Exit(1)
	}
	m, err := eg2.Open(*modelPath, opt)
	if err != nil {
		log.Error("opening the model", "error", err)
		os.Exit(1)
	}
	defer m.Close()
	started := time.Now()
	if err := m.Warm(); err != nil {
		log.Error("compiling the model", "error", err)
		os.Exit(1)
	}
	if err := m.SelfCheck(); err != nil {
		log.Error("the model failed its self check", "error", err)
		os.Exit(1)
	}
	log.Info("ready", "model", m.Describe(), "seconds", time.Since(started).Round(time.Millisecond).Seconds(),
		"alias", *alias)

	srv := &server.Server{Model: model{m}, Alias: *alias, Log: log, Created: time.Now()}
	if err := lands.ListenAndServe(ctx, *listen, srv.Handler()); err != nil {
		log.Error("serving", "error", err)
		os.Exit(1)
	}
}
