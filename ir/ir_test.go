package ir

import (
	"slices"
	"strings"
	"testing"
)

func TestShapes(t *testing.T) {
	g := New("t", NewPool())
	x := g.Parameter("x", F32, 1, -1, 512)
	w := g.F32s([]int64{768, 512}, make([]float32, 768*512)...)
	cases := []struct {
		v    Value
		want []int64
	}{
		{g.MatMul(x, w, false, true), []int64{1, -1, 768}},
		{g.Reshape(x, 0, 0, 2, 256), []int64{1, -1, 2, 256}},
		{g.Reshape(g.Parameter("s", F32, 2, 3, 4), 0, -1), []int64{2, 12}},
		{g.Transpose(g.Reshape(x, 0, 0, 2, 256), 0, 2, 1, 3), []int64{1, 2, -1, 256}},
		{g.Slice(x, -1, 0, 256), []int64{1, -1, 256}},
		{g.Concat(-1, g.Slice(x, -1, 0, 256), g.Slice(x, -1, 256, 512)), []int64{1, -1, 512}},
		{g.Add(x, g.Parameter("m", F32, 1, 1, 1)), []int64{1, -1, 512}},
		{g.MeanLast(x), []int64{1, -1, 1}},
		{g.Gather(g.Parameter("p", F32, 1, 7, 24, 512), 2, 3), []int64{1, 7, 512}},
		{g.MatMul(g.Parameter("q", F32, 1, 2, 2, 9, 256), g.Parameter("k", F32, 1, 2, 1, 9, 256), false, true),
			[]int64{1, 2, 2, 9, 9}},
	}
	for i, c := range cases {
		if !slices.Equal(c.v.Shape(), c.want) {
			t.Errorf("case %d: shape %v, want %v", i, c.v.Shape(), c.want)
		}
	}
	if g.Err != nil {
		t.Fatal(g.Err)
	}
	g.MatMul(x, g.F32s([]int64{3, 4}, make([]float32, 12)...), false, false)
	if g.Err == nil {
		t.Error("a MatMul of 512 by 3 went through")
	}
}

// The same constant in two graphs is stored once, so the graphs for every
// sequence length share one weights file.
func TestPoolShares(t *testing.T) {
	pool := NewPool()
	for _, l := range []int64{32, 64} {
		g := New("t", pool)
		x := g.Parameter("x", F32, 1, l, 4)
		g.Result("y", g.Mul(x, g.F32s([]int64{4}, 1, 2, 3, 4)))
		var b strings.Builder
		if err := g.Write(&b); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(b.String(), `offset="0" size="16"`) {
			t.Errorf("length %d: constant not at the start:\n%s", l, b.String())
		}
	}
	if n := len(pool.Bytes()); n != 16 {
		t.Errorf("pool of %d bytes", n)
	}
}
