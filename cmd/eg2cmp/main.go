// Command eg2cmp embeds a set of inputs with the OpenVINO model and compares
// each vector with what a llama-server answered for it, or with a file of
// vectors saved from one (-save, -ref), for machines without llama.cpp.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/daaku/eg2npu/eg2"
)

type refFile struct {
	Inputs  []string    `json:"inputs"`
	Vectors [][]float32 `json:"vectors"`
}

func llama(url string, inputs []string) ([][]float32, error) {
	var out [][]float32
	for _, in := range inputs {
		body, _ := json.Marshal(map[string]any{"input": []string{in}, "encoding_format": "float"})
		res, err := http.Post(url+"/v1/embeddings", "application/json", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		var r struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		err = json.NewDecoder(res.Body).Decode(&r)
		res.Body.Close()
		if err != nil || len(r.Data) != 1 {
			return nil, fmt.Errorf("llama-server answered %d: %v", res.StatusCode, err)
		}
		out = append(out, r.Data[0].Embedding)
	}
	return out, nil
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

func main() {
	model := flag.String("model", os.ExpandEnv("$HOME/.cache/eg2/embeddinggemma-2-Q8_0.gguf"), "GGUF")
	inputsPath := flag.String("inputs", "", "JSON list of input strings")
	llamaURL := flag.String("llama", "", "llama-server to compare with")
	refPath := flag.String("ref", "", "reference vectors (from -save) to compare with")
	save := flag.String("save", "", "save llama-server's vectors here and exit")
	device := flag.String("device", "CPU", "OpenVINO device")
	dynamic := flag.Bool("dynamic", false, "one dynamic graph instead of static lengths")
	weights := flag.String("weights", "f16", "f16 or q8")
	rscale := flag.Float64("rscale", 1, "residual scale")
	cache := flag.String("cache", os.ExpandEnv("$HOME/.cache/eg2npu"), "cache dir")
	repeat := flag.Int("repeat", 1, "timed runs per input")
	lengths := flag.String("lengths", "", "comma separated static lengths")
	flag.Parse()

	var ref refFile
	if *refPath != "" {
		data, err := os.ReadFile(*refPath)
		check(err)
		check(json.Unmarshal(data, &ref))
	} else {
		data, err := os.ReadFile(*inputsPath)
		check(err)
		check(json.Unmarshal(data, &ref.Inputs))
		ref.Vectors, err = llama(*llamaURL, ref.Inputs)
		check(err)
		if *save != "" {
			data, _ := json.Marshal(ref)
			check(os.WriteFile(*save, data, 0o644))
			return
		}
	}
	opt := eg2.Options{Device: *device, CacheDir: *cache, Dynamic: *dynamic, MaxTokens: 2048, Weights: eg2.Weights(*weights),
		ResidualScale: float32(*rscale), Log: func(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...) }}
	if *lengths != "" {
		for _, s := range strings.Split(*lengths, ",") {
			var n int
			fmt.Sscan(s, &n)
			opt.Lengths = append(opt.Lengths, n)
		}
	}
	check(os.MkdirAll(*cache, 0o755))
	m, err := eg2.Open(*model, opt)
	check(err)
	defer m.Close()
	started := time.Now()
	check(m.Warm())
	check(m.SelfCheck())
	fmt.Fprintf(os.Stderr, "warm: %v\n", time.Since(started).Round(time.Millisecond))
	var worst = 1.0
	var sum float64
	for i, in := range ref.Inputs {
		var r eg2.Result
		var times []time.Duration
		for range *repeat {
			t := time.Now()
			r, err = m.Embed(in)
			check(err)
			times = append(times, time.Since(t))
		}
		slices.Sort(times)
		c := cosine(r.Vector, ref.Vectors[i])
		worst = min(worst, c)
		sum += c
		fmt.Printf("%3d tokens=%4d len=%4d cos=%.6f p50=%v\n", i, r.Tokens, r.Length, c, times[len(times)/2].Round(10*time.Microsecond))
	}
	fmt.Printf("mean cos=%.6f worst=%.6f over %d inputs\n", sum/float64(len(ref.Inputs)), worst, len(ref.Inputs))
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
