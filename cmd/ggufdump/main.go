// Command ggufdump prints a GGUF file's metadata and tensor list.
package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/daaku/eg2npu/gguf"
)

func main() {
	g, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	keys := make([]string, 0, len(g.KV))
	for k := range g.KV {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := g.KV[k]
		if a, ok := v.([]any); ok {
			if len(a) > 8 {
				fmt.Printf("%s = [%d items] %v ...\n", k, len(a), a[:8])
				continue
			}
		}
		fmt.Printf("%s = %v\n", k, v)
	}
	for _, t := range g.Order {
		fmt.Printf("%-40s %-5v %v\n", t.Name, t.Type, t.Dims)
	}
}
