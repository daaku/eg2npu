// Command eg2tok prints the token ids of each file named, one line per file,
// in llama-tokenize's --ids format, for comparing the two tokenizers.
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/daaku/eg2npu/gguf"
	"github.com/daaku/eg2npu/tokenizer"
)

func main() {
	g, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	tok, err := tokenizer.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, name := range os.Args[2:] {
		data, err := os.ReadFile(name)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		ids := tok.Encode(string(data))
		parts := make([]string, len(ids))
		for i, id := range ids {
			parts[i] = fmt.Sprint(id)
		}
		fmt.Println("[" + strings.Join(parts, ", ") + "]")
	}
}
