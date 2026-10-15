package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Every half converts to a float and back to itself.
func TestHalfRoundTrip(t *testing.T) {
	for h := range 1 << 16 {
		f := HalfToFloat(uint16(h))
		if math.IsNaN(float64(f)) {
			if back := FloatToHalf(f); back&0x7c00 != 0x7c00 || back&0x3ff == 0 {
				t.Fatalf("%#04x: NaN came back as %#04x", h, back)
			}
			continue
		}
		if back := FloatToHalf(f); back != uint16(h) {
			t.Fatalf("%#04x -> %g -> %#04x", h, f, back)
		}
	}
}

// Rounding to half is to nearest, ties to even, and out of range is
// infinity.
func TestFloatToHalf(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{1, 0x3c00}, {-2, 0xc000}, {65504, 0x7bff}, {65520, 0x7c00}, {1e10, 0x7c00},
		{1 + 1.0/2048, 0x3c00},        // a tie, to the even 1
		{1 + 3.0/2048, 0x3c02},        // a tie, to the even 1+2/1024
		{1 + 1.0/2048 + 1e-7, 0x3c01}, // past the tie
		{5.960464477539063e-08, 0x0001},
		{2.9e-08, 0x0000},
	}
	for _, c := range cases {
		if got := FloatToHalf(c.f); got != c.want {
			t.Errorf("FloatToHalf(%g) = %#04x, want %#04x", c.f, got, c.want)
		}
	}
}

// A small GGUF written here: metadata of several types and a Q8_0, F16, BF16
// and F32 tensor, read back and dequantized.
func TestOpen(t *testing.T) {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(4)) // tensors
	w(uint64(3)) // metadata
	str("a.u32")
	w(uint32(4))
	w(uint32(7))
	str("a.f32")
	w(uint32(6))
	w(float32(0.5))
	str("a.arr")
	w(uint32(9))
	w(uint32(8))
	w(uint64(2))
	str("x")
	str("yz")
	type info struct {
		name string
		dims []uint64
		typ  Type
		size uint64
	}
	infos := []info{{"q", []uint64{32, 2}, Q8_0, 68}, {"h", []uint64{2}, F16, 4}, {"b", []uint64{2}, BF16, 4}, {"f", []uint64{1}, F32, 4}}
	var off uint64
	for _, in := range infos {
		str(in.name)
		w(uint32(len(in.dims)))
		for _, d := range in.dims {
			w(d)
		}
		w(uint32(in.typ))
		w(off)
		off += (in.size + 31) / 32 * 32
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	pad := func() {
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
	}
	for row := range 2 {
		w(FloatToHalf(0.25 * float32(row+1)))
		for j := range 32 {
			w(int8(j - 16))
		}
	}
	pad()
	w(FloatToHalf(1.5))
	w(FloatToHalf(-3))
	pad()
	w(uint16(math.Float32bits(2) >> 16))
	w(uint16(math.Float32bits(-0.5) >> 16))
	pad()
	w(float32(42))
	path := filepath.Join(t.TempDir(), "x.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if v, ok := g.Uint("a.u32"); !ok || v != 7 {
		t.Errorf("a.u32 = %v", v)
	}
	if v, ok := g.Float("a.f32"); !ok || v != 0.5 {
		t.Errorf("a.f32 = %v", v)
	}
	if a, _ := g.KV["a.arr"].([]any); len(a) != 2 || a[1] != "yz" {
		t.Errorf("a.arr = %v", g.KV["a.arr"])
	}
	q, _ := g.Tensor("q")
	row := make([]float32, 32)
	g.Row(q, 1, row)
	if row[0] != -16*0.5 || row[31] != 15*0.5 {
		t.Errorf("q row 1 = %v", row)
	}
	for _, c := range []struct {
		name string
		want []float32
	}{{"h", []float32{1.5, -3}}, {"b", []float32{2, -0.5}}, {"f", []float32{42}}} {
		tt, err := g.Tensor(c.name)
		if err != nil {
			t.Fatal(err)
		}
		got := g.Float32(tt)
		for i := range c.want {
			if got[i] != c.want[i] {
				t.Errorf("%s = %v, want %v", c.name, got, c.want)
			}
		}
	}
	if _, err := g.Tensor("nope"); err == nil {
		t.Error("found a tensor that is not there")
	}
	// A file cut short is refused rather than read past its end.
	if err := os.WriteFile(path, b.Bytes()[:b.Len()-10], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Error("opened a truncated file")
	}
}
