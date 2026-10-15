package server

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fake embeds a text as its length and a token list as its first id, so a
// test can see which input went where.
type fake struct{}

func (fake) Embed(text string) ([]float32, int, error) {
	if text == "fail" {
		return nil, 0, errors.New("input (3000 tokens) is too large to process")
	}
	return []float32{float32(len(text)), 1}, len(text) + 2, nil
}

func (fake) EmbedTokens(ids []int32) ([]float32, int, error) {
	return []float32{float32(ids[0]), 2}, len(ids), nil
}

func (fake) MaxTokens() int { return 2048 }

func serve(t *testing.T) *httptest.Server {
	s := &Server{Model: fake{}, Alias: "embeddinggemma2", Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func post(t *testing.T, ts *httptest.Server, body string) (int, map[string]any) {
	t.Helper()
	res, err := http.Post(ts.URL+"/v1/embeddings", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return res.StatusCode, out
}

func TestEmbeddings(t *testing.T) {
	ts := serve(t)
	cases := []struct {
		body  string
		first []float64 // each item's first component
	}{
		{`{"input":"abc"}`, []float64{3}},
		{`{"input":["a","bb",""]}`, []float64{1, 2, 0}},
		{`{"input":[5,6,7]}`, []float64{5}},
		{`{"input":[[9],[8,1]]}`, []float64{9, 8}},
		{`{"content":"abcd"}`, []float64{4}},
		{`{"input":"abc","encoding_format":"float"}`, []float64{3}},
	}
	for _, c := range cases {
		status, out := post(t, ts, c.body)
		data, _ := out["data"].([]any)
		if status != http.StatusOK || len(data) != len(c.first) {
			t.Errorf("%s: %d %v", c.body, status, out)
			continue
		}
		for i, item := range data {
			m := item.(map[string]any)
			v := m["embedding"].([]any)
			if m["index"].(float64) != float64(i) || v[0].(float64) != c.first[i] {
				t.Errorf("%s: item %d is %v", c.body, i, m)
			}
		}
	}
}

// base64 is little endian float32s, which is what khoj asks for.
func TestBase64(t *testing.T) {
	_, out := post(t, serve(t), `{"input":["abcde"],"encoding_format":"base64"}`)
	item := out["data"].([]any)[0].(map[string]any)
	raw, err := base64.StdEncoding.DecodeString(item["embedding"].(string))
	if err != nil || len(raw) != 8 {
		t.Fatalf("%v %v", item, err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(raw)); v != 5 {
		t.Errorf("first component %g", v)
	}
	if usage := out["usage"].(map[string]any); usage["prompt_tokens"].(float64) != 7 {
		t.Errorf("usage %v", usage)
	}
}

func TestErrors(t *testing.T) {
	ts := serve(t)
	cases := []struct {
		body   string
		status int
		says   string
	}{
		{`{nope`, 400, "not JSON"},
		{`{}`, 400, "must be provided"},
		{`{"input":[]}`, 400, "must not be empty"},
		{`{"input":"a","encoding_format":"int8"}`, 400, "float or base64"},
		{`{"input":[{"content":[{"type":"image_url","image_url":{"url":"data:,"}}]}]}`, 400, "only text"},
		{`{"input":{"content":"x"}}`, 400, "only text"},
		{`{"input":[true]}`, 400, "neither a string"},
		{`{"input":["ok","fail"]}`, 500, "too large"},
	}
	for _, c := range cases {
		status, out := post(t, ts, c.body)
		e, _ := out["error"].(map[string]any)
		msg, _ := e["message"].(string)
		if status != c.status || !strings.Contains(msg, c.says) {
			t.Errorf("%s: %d %v", c.body, status, out)
		}
	}
}

// khojd reads the model's name from /v1/models and each slot's context from
// /props.
func TestModelsAndProps(t *testing.T) {
	ts := serve(t)
	get := func(path string) map[string]any {
		res, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		var out map[string]any
		json.NewDecoder(res.Body).Decode(&out)
		return out
	}
	if id := get("/v1/models")["data"].([]any)[0].(map[string]any)["id"]; id != "embeddinggemma2" {
		t.Errorf("model %v", id)
	}
	props := get("/props")
	if props["default_generation_settings"].(map[string]any)["n_ctx"].(float64) != 2048 || props["total_slots"].(float64) != 1 {
		t.Errorf("props %v", props)
	}
	if get("/health")["status"] != "ok" {
		t.Error("not healthy")
	}
}
