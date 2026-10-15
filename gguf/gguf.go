// Package gguf reads the GGUF files llama.cpp loads: the metadata, and the
// tensors dequantized to float32. The file is mapped rather than read, so a
// tensor that is never asked for costs nothing.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"syscall"
)

// Type is a ggml tensor type.
type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q8_0 Type = 8
	BF16 Type = 30
)

func (t Type) String() string {
	switch t {
	case F32:
		return "F32"
	case F16:
		return "F16"
	case Q8_0:
		return "Q8_0"
	case BF16:
		return "BF16"
	}
	return fmt.Sprintf("type(%d)", uint32(t))
}

// Tensor is one tensor's place in the file. Dims are in ggml order: Dims[0] is
// the innermost (contiguous) one, so a linear layer's weight is {in, out}.
type Tensor struct {
	Name   string
	Dims   []int
	Type   Type
	offset uint64
}

// Elements is the number of values in the tensor.
func (t *Tensor) Elements() int {
	n := 1
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// File is an open GGUF file.
type File struct {
	KV      map[string]any
	Tensors map[string]*Tensor
	// Order is the tensors in file order.
	Order []*Tensor
	data  []byte
	base  uint64
}

// Open maps a GGUF file and reads its header.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("gguf: mapping %s: %w", path, err)
	}
	g := &File{KV: map[string]any{}, Tensors: map[string]*Tensor{}, data: data}
	if err := g.parse(); err != nil {
		syscall.Munmap(data)
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}
	return g, nil
}

// Close unmaps the file. Slices returned by Raw are invalid after it.
func (g *File) Close() error {
	if g.data == nil {
		return nil
	}
	err := syscall.Munmap(g.data)
	g.data = nil
	return err
}

var errShort = errors.New("truncated file")

type reader struct {
	b   []byte
	off uint64
	err error
}

func (r *reader) take(n uint64) []byte {
	if r.err != nil {
		return nil
	}
	if r.off+n > uint64(len(r.b)) {
		r.err = errShort
		return nil
	}
	p := r.b[r.off : r.off+n]
	r.off += n
	return p
}

func (r *reader) u32() uint32 {
	if p := r.take(4); p != nil {
		return binary.LittleEndian.Uint32(p)
	}
	return 0
}

func (r *reader) u64() uint64 {
	if p := r.take(8); p != nil {
		return binary.LittleEndian.Uint64(p)
	}
	return 0
}

func (r *reader) str() string { return string(r.take(r.u64())) }

func (r *reader) value(kind uint32) any {
	switch kind {
	case 0:
		if p := r.take(1); p != nil {
			return uint8(p[0])
		}
	case 1:
		if p := r.take(1); p != nil {
			return int8(p[0])
		}
	case 2:
		if p := r.take(2); p != nil {
			return binary.LittleEndian.Uint16(p)
		}
	case 3:
		if p := r.take(2); p != nil {
			return int16(binary.LittleEndian.Uint16(p))
		}
	case 4:
		return r.u32()
	case 5:
		return int32(r.u32())
	case 6:
		return math.Float32frombits(r.u32())
	case 7:
		if p := r.take(1); p != nil {
			return p[0] != 0
		}
	case 8:
		return r.str()
	case 9:
		elem := r.u32()
		n := r.u64()
		if n > uint64(len(r.b)) {
			r.err = errShort
			return nil
		}
		out := make([]any, 0, n)
		for range n {
			out = append(out, r.value(elem))
			if r.err != nil {
				return nil
			}
		}
		return out
	case 10:
		return r.u64()
	case 11:
		return int64(r.u64())
	case 12:
		return math.Float64frombits(r.u64())
	default:
		r.err = fmt.Errorf("unknown metadata type %d", kind)
	}
	return nil
}

func (g *File) parse() error {
	r := &reader{b: g.data}
	if string(r.take(4)) != "GGUF" {
		return errors.New("not a GGUF file")
	}
	if v := r.u32(); v != 3 && r.err == nil {
		return fmt.Errorf("GGUF version %d, want 3", v)
	}
	nt, nkv := r.u64(), r.u64()
	for range nkv {
		key := r.str()
		g.KV[key] = r.value(r.u32())
		if r.err != nil {
			return r.err
		}
	}
	for range nt {
		t := &Tensor{Name: r.str()}
		nd := r.u32()
		for range nd {
			t.Dims = append(t.Dims, int(r.u64()))
		}
		t.Type = Type(r.u32())
		t.offset = r.u64()
		if r.err != nil {
			return r.err
		}
		g.Tensors[t.Name] = t
		g.Order = append(g.Order, t)
	}
	align := uint64(32)
	if a, ok := g.KV["general.alignment"].(uint32); ok {
		align = uint64(a)
	}
	g.base = (r.off + align - 1) / align * align
	for _, t := range g.Order {
		n, err := t.bytes()
		if err != nil {
			return fmt.Errorf("%s: %w", t.Name, err)
		}
		if g.base+t.offset+n > uint64(len(g.data)) {
			return fmt.Errorf("%s: %w", t.Name, errShort)
		}
	}
	return nil
}

func (t *Tensor) bytes() (uint64, error) {
	n := uint64(t.Elements())
	switch t.Type {
	case F32:
		return n * 4, nil
	case F16, BF16:
		return n * 2, nil
	case Q8_0:
		if n%32 != 0 {
			return 0, fmt.Errorf("Q8_0 tensor of %d values", n)
		}
		return n / 32 * 34, nil
	}
	return 0, fmt.Errorf("unsupported tensor type %v", t.Type)
}

// Raw is the tensor's bytes as stored, aliasing the mapped file.
func (g *File) Raw(t *Tensor) []byte {
	n, _ := t.bytes() // checked by parse
	start := g.base + t.offset
	return g.data[start : start+n]
}

// Tensor finds a tensor by name.
func (g *File) Tensor(name string) (*Tensor, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	return t, nil
}

// Float32 dequantizes a whole tensor.
func (g *File) Float32(t *Tensor) []float32 {
	out := make([]float32, t.Elements())
	Dequantize(t.Type, g.Raw(t), out)
	return out
}

// Row dequantizes row i of a 2D tensor into out, which holds Dims[0] values.
// It is how a token's embedding is looked up without touching the rest.
func (g *File) Row(t *Tensor, i int, out []float32) {
	raw := g.Raw(t)
	n := t.Dims[0]
	switch t.Type {
	case F32:
		raw = raw[i*n*4 : (i+1)*n*4]
	case F16, BF16:
		raw = raw[i*n*2 : (i+1)*n*2]
	case Q8_0:
		raw = raw[i*n/32*34 : (i+1)*n/32*34]
	}
	Dequantize(t.Type, raw, out[:n])
}

// Dequantize converts raw values of type t into out. The arithmetic is
// ggml's: a Q8_0 value is its int8 times the block's float16 scale, in
// float32.
func Dequantize(t Type, raw []byte, out []float32) {
	switch t {
	case F32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case F16:
		for i := range out {
			out[i] = HalfToFloat(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case BF16:
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case Q8_0:
		for b := 0; b*32 < len(out); b++ {
			block := raw[b*34 : b*34+34]
			d := HalfToFloat(binary.LittleEndian.Uint16(block))
			for j := range 32 {
				out[b*32+j] = float32(int8(block[2+j])) * d
			}
		}
	}
}

// HalfToFloat converts an IEEE half precision value.
func HalfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
}

// FloatToHalf converts to IEEE half precision, rounding to nearest even.
func FloatToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127 + 15
	frac := b & 0x7fffff
	switch {
	case b&0x7fffffff == 0:
		return sign
	case b>>23&0xff == 0xff: // inf, nan
		if frac != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		v := frac >> shift
		rem := frac & (1<<shift - 1)
		if rem > half || (rem == half && v&1 == 1) {
			v++
		}
		return sign | uint16(v)
	}
	v := uint32(exp)<<10 | frac>>13
	rem := frac & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && v&1 == 1) {
		v++
	}
	return sign | uint16(v)
}

// Uint reads an integer metadata value of any width.
func (g *File) Uint(key string) (int, bool) { return Int(g.KV[key]) }

// Int converts a metadata integer of any width, as found in arrays too.
func Int(value any) (int, bool) {
	switch v := value.(type) {
	case uint8:
		return int(v), true
	case uint16:
		return int(v), true
	case uint32:
		return int(v), true
	case int32:
		return int(v), true
	case uint64:
		return int(v), true
	case int64:
		return int(v), true
	}
	return 0, false
}

// Float reads a float metadata value.
func (g *File) Float(key string) (float64, bool) {
	switch v := g.KV[key].(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	}
	return 0, false
}
