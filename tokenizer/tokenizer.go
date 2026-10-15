// Package tokenizer is the Gemma 4 tokenizer as llama.cpp runs it, read from
// the vocabulary in a GGUF file. Matching llama.cpp token for token is the
// point: khoj's index was embedded by llama-server, and a query tokenized any
// other way lands somewhere else in the vector space.
//
// The steps are llama.cpp's (src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_GEMMA4):
//
//  1. Special tokens (control, user defined and unknown) are cut out of the
//     text first, longest text first, as llama-server does with parse_special.
//  2. In the rest, spaces become "▁" and the text is split into runs of
//     newlines and runs of everything else.
//  3. A run of newlines that is itself a token is that token. Anything else is
//     split into UTF-8 characters and merged by BPE rank, lowest rank first and
//     leftmost first on a tie.
//  4. A piece that is not in the vocabulary is spelled with byte tokens,
//     "<0xXX>".
//  5. BOS goes in front and EOS behind, as the GGUF asks for.
package tokenizer

import (
	"container/heap"
	"fmt"
	"slices"
	"strings"

	"github.com/daaku/eg2npu/gguf"
)

const (
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
)

type pair struct{ left, right string }

// Tokenizer turns text into token ids.
type Tokenizer struct {
	tokens  []string
	ids     map[string]int32
	ranks   map[pair]int
	special []int32 // longest text first
	bos     int32
	eos     int32
	addBOS  bool
	addEOS  bool
}

// FromGGUF reads the tokenizer out of a GGUF file's metadata.
func FromGGUF(g *gguf.File) (*Tokenizer, error) {
	if model, _ := g.KV["tokenizer.ggml.model"].(string); model != "gemma4" {
		return nil, fmt.Errorf("tokenizer: model %q, want gemma4", model)
	}
	rawTokens, _ := g.KV["tokenizer.ggml.tokens"].([]any)
	rawTypes, _ := g.KV["tokenizer.ggml.token_type"].([]any)
	rawMerges, _ := g.KV["tokenizer.ggml.merges"].([]any)
	if len(rawTokens) == 0 || len(rawTypes) != len(rawTokens) || len(rawMerges) == 0 {
		return nil, fmt.Errorf("tokenizer: GGUF lacks tokens, token types or merges")
	}
	t := &Tokenizer{
		tokens: make([]string, len(rawTokens)),
		ids:    make(map[string]int32, len(rawTokens)),
		ranks:  make(map[pair]int, len(rawMerges)),
		// llama.cpp forces add_bos on for Gemma 4 whatever the file says.
		addBOS: true,
	}
	for i, v := range rawTokens {
		s, _ := v.(string)
		t.tokens[i] = s
		// The first id wins a duplicate, as in llama.cpp's token_to_id map.
		if _, dup := t.ids[s]; !dup {
			t.ids[s] = int32(i)
		}
		typ, _ := rawTypes[i].(int32)
		if (typ == typeControl || typ == typeUserDefined || typ == typeUnknown) && s != "" {
			t.special = append(t.special, int32(i))
		}
	}
	slices.SortStableFunc(t.special, func(a, b int32) int { return len(t.tokens[b]) - len(t.tokens[a]) })
	for i, v := range rawMerges {
		s, _ := v.(string)
		// Split at the first space after the first byte, as llama.cpp does.
		if pos := strings.IndexByte(s[min(1, len(s)):], ' '); pos >= 0 {
			pos++
			t.ranks[pair{s[:pos], s[pos+1:]}] = i
		} else {
			t.ranks[pair{}] = i
		}
	}
	bos, okB := g.Uint("tokenizer.ggml.bos_token_id")
	eos, okE := g.Uint("tokenizer.ggml.eos_token_id")
	if !okB || !okE {
		return nil, fmt.Errorf("tokenizer: GGUF lacks BOS or EOS ids")
	}
	t.bos, t.eos = int32(bos), int32(eos)
	if v, ok := g.KV["tokenizer.ggml.add_eos_token"].(bool); ok {
		t.addEOS = v
	}
	return t, nil
}

// Size is the vocabulary size.
func (t *Tokenizer) Size() int { return len(t.tokens) }

// Token is the text of token id.
func (t *Tokenizer) Token(id int32) string { return t.tokens[id] }

type fragment struct {
	text  string
	token int32 // -1 for text
}

// Encode tokenizes text the way llama-server tokenizes an embedding input:
// with BOS and EOS added and special tokens in the text parsed.
func (t *Tokenizer) Encode(text string) []int32 {
	var out []int32
	if t.addBOS {
		out = append(out, t.bos)
	}
	for _, f := range t.partition(text) {
		if f.token >= 0 {
			out = append(out, f.token)
			continue
		}
		out = t.bpe(strings.ReplaceAll(f.text, " ", "▁"), out)
	}
	if t.addEOS {
		out = append(out, t.eos)
	}
	return out
}

// partition cuts the special tokens out of the text, one special token at a
// time over every text fragment left, as llama.cpp's tokenizer_st_partition
// does. No Gemma 4 token strips the spaces around it.
func (t *Tokenizer) partition(text string) []fragment {
	frags := []fragment{{text: text, token: -1}}
	if text == "" {
		return nil
	}
	for _, id := range t.special {
		special := t.tokens[id]
		var next []fragment
		changed := false
		for _, f := range frags {
			if f.token >= 0 || !strings.Contains(f.text, special) {
				next = append(next, f)
				continue
			}
			changed = true
			rest := f.text
			for {
				i := strings.Index(rest, special)
				if i < 0 {
					if rest != "" {
						next = append(next, fragment{text: rest, token: -1})
					}
					break
				}
				if i > 0 {
					next = append(next, fragment{text: rest[:i], token: -1})
				}
				next = append(next, fragment{token: id})
				rest = rest[i+len(special):]
			}
		}
		if changed {
			frags = next
		}
	}
	return frags
}

// words splits on the regex "[^\n]+|[\n]+".
func words(text string) []string {
	var out []string
	for len(text) > 0 {
		nl := text[0] == '\n'
		i := 1
		for i < len(text) && (text[i] == '\n') == nl {
			i++
		}
		out = append(out, text[:i])
		text = text[i:]
	}
	return out
}

func utf8Len(b byte) int {
	return [16]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}[b>>4]
}

type symbol struct {
	start, n   int
	prev, next int
}

type bigram struct {
	left, right int
	text        string
	rank        int
}

type queue []bigram

func (q queue) Len() int { return len(q) }
func (q queue) Less(i, j int) bool {
	return q[i].rank < q[j].rank || (q[i].rank == q[j].rank && q[i].left < q[j].left)
}
func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)   { *q = append(*q, x.(bigram)) }
func (q *queue) Pop() any {
	old := *q
	b := old[len(old)-1]
	*q = old[:len(old)-1]
	return b
}

func (t *Tokenizer) bpe(text string, out []int32) []int32 {
	for _, word := range words(text) {
		var syms []symbol
		offset := 0
		if strings.Trim(word, "\n") == "" {
			if _, ok := t.ids[word]; ok {
				syms = append(syms, symbol{start: 0, n: len(word), prev: -1, next: -1})
				offset = len(word)
			}
		}
		for offset < len(word) {
			n := min(len(word)-offset, utf8Len(word[offset]))
			idx := len(syms)
			next := idx + 1
			if offset+n == len(word) {
				next = -1
			}
			syms = append(syms, symbol{start: offset, n: n, prev: idx - 1, next: next})
			offset += n
		}
		var q queue
		text := func(s symbol) string { return word[s.start : s.start+s.n] }
		add := func(left, right int) {
			if left < 0 || right < 0 {
				return
			}
			l, r := text(syms[left]), text(syms[right])
			rank, ok := t.ranks[pair{l, r}]
			if !ok {
				return
			}
			heap.Push(&q, bigram{left: left, right: right, text: l + r, rank: rank})
		}
		for i := 1; i < len(syms); i++ {
			add(i-1, i)
		}
		for q.Len() > 0 {
			b := heap.Pop(&q).(bigram)
			left, right := &syms[b.left], &syms[b.right]
			if left.n == 0 || right.n == 0 || text(*left)+text(*right) != b.text {
				continue
			}
			left.n += right.n
			right.n = 0
			left.next = right.next
			if right.next >= 0 {
				syms[right.next].prev = b.left
			}
			add(left.prev, b.left)
			add(b.left, left.next)
		}
		for _, s := range syms {
			if s.n == 0 {
				continue
			}
			piece := text(s)
			if id, ok := t.ids[piece]; ok {
				out = append(out, id)
				continue
			}
			for i := 0; i < len(piece); i++ {
				if id, ok := t.ids[fmt.Sprintf("<0x%02X>", piece[i])]; ok {
					out = append(out, id)
				}
			}
		}
	}
	return out
}
