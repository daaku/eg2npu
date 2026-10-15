package tokenizer

import (
	"slices"
	"testing"

	"github.com/daaku/eg2npu/internal/testmodel"
)

// The ids are llama-tokenize's, for prose, code, several scripts, special
// tokens written in the text, odd whitespace and byte fallbacks.
func TestMatchesLlama(t *testing.T) {
	tok, err := FromGGUF(testmodel.Open(t))
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range testmodel.Load(t).Tokenize {
		if got := tok.Encode(c.Text); !slices.Equal(got, c.IDs) {
			t.Errorf("input %d %.40q:\n got %v\nwant %v", i, c.Text, got, c.IDs)
		}
	}
	// An empty text is BOS and EOS, as llama-server embeds it.
	if got := tok.Encode(""); !slices.Equal(got, []int32{2, 1}) {
		t.Errorf("empty: %v", got)
	}
}

func TestWords(t *testing.T) {
	got := words("a\n\nb c\n")
	if want := []string{"a", "\n\n", "b c", "\n"}; !slices.Equal(got, want) {
		t.Errorf("words = %q, want %q", got, want)
	}
}

func BenchmarkEncode(b *testing.B) {
	tok, err := FromGGUF(testmodel.Open(b))
	if err != nil {
		b.Fatal(err)
	}
	text := testmodel.Load(b).Embed[9].Text
	for b.Loop() {
		tok.Encode(text)
	}
}
