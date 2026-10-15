// Package server answers the part of llama-server's HTTP API that an
// embedding client uses: POST /v1/embeddings, GET /v1/models, GET /props and
// GET /health. It is text only; an image or audio input is refused.
package server

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"time"
)

// Embedder is the model: one text or one token list in, one normalized
// vector out.
type Embedder interface {
	Embed(text string) ([]float32, int, error)
	EmbedTokens(ids []int32) ([]float32, int, error)
	MaxTokens() int
}

// Server is the HTTP API over an Embedder.
type Server struct {
	Model Embedder
	// Alias is the model's name in /v1/models, which khojd stores with its
	// index: it must be the --alias llama-server was started with
	// (embeddinggemma2), or khojd embeds everything again.
	Alias string
	Log   *slog.Logger
	// Created is reported in /v1/models.
	Created time.Time
}

// Handler routes the API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/embeddings", s.embeddings)
	mux.HandleFunc("POST /embeddings", s.embeddings)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /models", s.models)
	mux.HandleFunc("GET /props", s.props)
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/health", s.health)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// writeError answers in llama-server's error shape.
func writeError(w http.ResponseWriter, status int, kind, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"code": status, "message": message, "type": kind}})
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.Alias, "object": "model", "created": s.Created.Unix(), "owned_by": "eg2npu",
		}},
	})
}

// props reports one slot holding the longest input, the numbers khojd checks
// to see whether its inputs fit.
func (s *Server) props(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"default_generation_settings": map[string]any{"n_ctx": s.Model.MaxTokens()},
		"total_slots":                 1,
		"model_alias":                 s.Alias,
	})
}

type input struct {
	text   string
	tokens []int32
}

var errNotText = errors.New("only text inputs are supported: this server runs EmbeddingGemma 2's text encoder alone")

// parseInputs reads llama-server's input shapes: a string, a list of
// strings, a list of token ids, or a list of those.
func parseInputs(raw json.RawMessage) ([]input, error) {
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []input{{text: one}}, nil
	}
	var ids []int32
	if json.Unmarshal(raw, &ids) == nil && len(ids) > 0 {
		return []input{{tokens: ids}}, nil
	}
	var list []json.RawMessage
	if err := json.Unmarshal(raw, &list); err != nil {
		var obj map[string]json.RawMessage
		if json.Unmarshal(raw, &obj) == nil {
			return nil, errNotText
		}
		return nil, errors.New("\"input\" must be a string, a list of strings or a list of token ids")
	}
	out := make([]input, 0, len(list))
	for _, item := range list {
		var text string
		if json.Unmarshal(item, &text) == nil {
			out = append(out, input{text: text})
			continue
		}
		var ids []int32
		if json.Unmarshal(item, &ids) == nil {
			out = append(out, input{tokens: ids})
			continue
		}
		var obj map[string]json.RawMessage
		if json.Unmarshal(item, &obj) == nil {
			return nil, errNotText
		}
		return nil, fmt.Errorf("input %d is neither a string nor a list of token ids", len(out))
	}
	return out, nil
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	var body struct {
		Input          json.RawMessage `json:"input"`
		Content        json.RawMessage `json:"content"`
		EncodingFormat string          `json:"encoding_format"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<20)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "the body is not JSON: "+err.Error())
		return
	}
	raw := body.Input
	if raw == nil {
		raw = body.Content
	}
	if raw == nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "\"input\" or \"content\" must be provided")
		return
	}
	useBase64 := false
	switch body.EncodingFormat {
	case "", "float":
	case "base64":
		useBase64 = true
	default:
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"The format to return the embeddings in. Can be either float or base64")
		return
	}
	inputs, err := parseInputs(raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "\"input\" must not be empty")
		return
	}
	data := make([]map[string]any, 0, len(inputs))
	total := 0
	for i, in := range inputs {
		var vector []float32
		var n int
		if in.tokens != nil {
			vector, n, err = s.Model.EmbedTokens(in.tokens)
		} else {
			vector, n, err = s.Model.Embed(in.text)
		}
		if err != nil {
			s.Log.Warn("embedding failed", "input", i, "error", err)
			writeError(w, http.StatusInternalServerError, "server_error", err.Error())
			return
		}
		total += n
		item := map[string]any{"index": i, "object": "embedding"}
		if useBase64 {
			buf := make([]byte, 4*len(vector))
			for j, v := range vector {
				binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(v))
			}
			item["embedding"] = base64.StdEncoding.EncodeToString(buf)
			item["encoding_format"] = "base64"
		} else {
			item["embedding"] = vector
		}
		data = append(data, item)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model":  s.Alias,
		"object": "list",
		"usage":  map[string]int{"prompt_tokens": total, "total_tokens": total},
		"data":   data,
	})
	s.Log.Debug("embedded", "inputs", len(inputs), "tokens", total,
		"ms", float64(time.Since(started).Microseconds())/1000)
}
