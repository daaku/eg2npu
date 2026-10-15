// Package ir writes OpenVINO IR (version 11): an XML graph and a binary file
// of constants. It knows only the operations the EmbeddingGemma 2 graph uses,
// and tracks shapes so every port carries its dimensions. A dimension of -1 is
// dynamic.
//
// Constants go through a Pool shared by several graphs: the same bytes get the
// same offset, so graphs that differ only in their input shapes (one per
// sequence length, for devices that need static shapes) share one weights file.
package ir

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Type is an element type, in the IR's spelling.
type Type string

const (
	F32  Type = "f32"
	F16  Type = "f16"
	I64  Type = "i64"
	I32  Type = "i32"
	I8   Type = "i8"
	Bool Type = "boolean"
)

func (t Type) precision() string {
	switch t {
	case F32:
		return "FP32"
	case F16:
		return "FP16"
	case I64:
		return "I64"
	case I32:
		return "I32"
	case I8:
		return "I8"
	case Bool:
		return "BOOL"
	}
	return strings.ToUpper(string(t))
}

func (t Type) size() int {
	switch t {
	case F32, I32:
		return 4
	case F16:
		return 2
	case I64:
		return 8
	}
	return 1
}

// Pool holds the constants' bytes, deduplicated by content.
type Pool struct {
	buf  bytes.Buffer
	seen map[[32]byte]int
}

// NewPool is an empty constant pool.
func NewPool() *Pool { return &Pool{seen: map[[32]byte]int{}} }

func (p *Pool) add(data []byte) int {
	sum := sha256.Sum256(data)
	if off, ok := p.seen[sum]; ok {
		return off
	}
	// Keep every constant 64 byte aligned, which some plugins map directly.
	for p.buf.Len()%64 != 0 {
		p.buf.WriteByte(0)
	}
	off := p.buf.Len()
	p.buf.Write(data)
	p.seen[sum] = off
	return off
}

// WriteTo writes the weights file.
func (p *Pool) WriteTo(w io.Writer) (int64, error) { return p.buf.WriteTo(w) }

// Bytes is the weights file's contents.
func (p *Pool) Bytes() []byte { return p.buf.Bytes() }

type node struct {
	id      int
	name    string
	kind    string
	version string
	attrs   [][2]string
	inputs  []Value
	outputs []port
}

type port struct {
	t     Type
	shape []int64
	names string
}

// Value is an output of a node.
type Value struct {
	n    *node
	port int
}

// Shape is the value's shape; -1 is a dynamic dimension.
func (v Value) Shape() []int64 { return v.n.outputs[v.port].shape }

// Type is the value's element type.
func (v Value) Type() Type { return v.n.outputs[v.port].t }

// Graph is a model being built.
type Graph struct {
	Name  string
	pool  *Pool
	nodes []*node
	// Err is the first error building the graph met; later calls do nothing
	// useful once it is set, and Write reports it.
	Err error
}

// New starts a graph whose constants go to pool.
func New(name string, pool *Pool) *Graph { return &Graph{Name: name, pool: pool} }

func (g *Graph) fail(format string, args ...any) {
	if g.Err == nil {
		g.Err = fmt.Errorf(format, args...)
	}
}

func (g *Graph) add(kind, version string, attrs [][2]string, inputs []Value, outputs ...port) *node {
	n := &node{id: len(g.nodes), kind: kind, version: version, attrs: attrs, inputs: inputs, outputs: outputs}
	n.name = fmt.Sprintf("%s_%d", strings.ToLower(kind), n.id)
	g.nodes = append(g.nodes, n)
	return n
}

func (g *Graph) one(kind, version string, attrs [][2]string, t Type, shape []int64, inputs ...Value) Value {
	return Value{n: g.add(kind, version, attrs, inputs, port{t: t, shape: shape})}
}

// Named renames a value's node, which is how outputs and debugging points
// are found again.
func (g *Graph) Named(v Value, name string) Value {
	v.n.name = name
	return v
}

func dims(shape []int64) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		if d < 0 {
			parts[i] = "?"
		} else {
			parts[i] = strconv.FormatInt(d, 10)
		}
	}
	return strings.Join(parts, ",")
}

// Parameter is a model input.
func (g *Graph) Parameter(name string, t Type, shape ...int64) Value {
	v := g.one("Parameter", "opset1", [][2]string{{"shape", dims(shape)}, {"element_type", string(t)}}, t, shape)
	v.n.name = name
	v.n.outputs[0].names = name
	return v
}

// Result marks a model output.
func (g *Graph) Result(name string, v Value) {
	v.n.outputs[v.port].names = name
	r := g.add("Result", "opset1", nil, []Value{v})
	r.name = name + "/sink"
}

// Const adds a constant from raw little endian bytes.
func (g *Graph) Const(t Type, shape []int64, data []byte) Value {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	if int64(len(data)) != n*int64(t.size()) {
		g.fail("ir: constant of shape %v and type %s has %d bytes", shape, t, len(data))
	}
	off := g.pool.add(data)
	return g.one("Const", "opset1", [][2]string{
		{"element_type", string(t)}, {"shape", dims(shape)},
		{"offset", strconv.Itoa(off)}, {"size", strconv.Itoa(len(data))},
	}, t, slices.Clone(shape))
}

// F32s is a float32 constant.
func (g *Graph) F32s(shape []int64, values ...float32) Value {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return g.Const(F32, shape, b)
}

// Scalar is a float32 scalar constant.
func (g *Graph) Scalar(v float32) Value { return g.F32s(nil, v) }

// I64s is a 1D int64 constant.
func (g *Graph) I64s(values ...int64) Value {
	b := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	return g.Const(I64, []int64{int64(len(values))}, b)
}

// I64Scalar is an int64 scalar constant.
func (g *Graph) I64Scalar(v int64) Value {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(v))
	return g.Const(I64, nil, b)
}

// Convert changes the element type.
func (g *Graph) Convert(v Value, t Type) Value {
	return g.one("Convert", "opset1", [][2]string{{"destination_type", string(t)}}, t, v.Shape(), v)
}

func broadcast(a, b []int64) ([]int64, error) {
	n := max(len(a), len(b))
	out := make([]int64, n)
	for i := range n {
		da, db := int64(1), int64(1)
		if j := i - (n - len(a)); j >= 0 {
			da = a[j]
		}
		if j := i - (n - len(b)); j >= 0 {
			db = b[j]
		}
		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		case da < 0:
			out[i] = db
		case db < 0:
			out[i] = da
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", a, b)
		}
	}
	return out, nil
}

func (g *Graph) binary(kind string, a, b Value) Value {
	shape, err := broadcast(a.Shape(), b.Shape())
	if err != nil {
		g.fail("ir: %s: %w", kind, err)
	}
	return g.one(kind, "opset1", [][2]string{{"auto_broadcast", "numpy"}}, a.Type(), shape, a, b)
}

// Add, Sub, Mul, Div and Pow are elementwise with numpy broadcasting.
func (g *Graph) Add(a, b Value) Value { return g.binary("Add", a, b) }
func (g *Graph) Sub(a, b Value) Value { return g.binary("Subtract", a, b) }
func (g *Graph) Mul(a, b Value) Value { return g.binary("Multiply", a, b) }
func (g *Graph) Div(a, b Value) Value { return g.binary("Divide", a, b) }
func (g *Graph) Pow(a, b Value) Value { return g.binary("Power", a, b) }

// Sqrt is the elementwise square root.
func (g *Graph) Sqrt(v Value) Value { return g.one("Sqrt", "opset1", nil, v.Type(), v.Shape(), v) }

// Gelu is GELU with the tanh approximation, as ggml_gelu computes it.
func (g *Graph) Gelu(v Value) Value {
	return g.one("Gelu", "opset7", [][2]string{{"approximation_mode", "TANH"}}, v.Type(), v.Shape(), v)
}

// Softmax over axis.
func (g *Graph) Softmax(v Value, axis int64) Value {
	return g.one("SoftMax", "opset8", [][2]string{{"axis", strconv.FormatInt(axis, 10)}}, v.Type(), v.Shape(), v)
}

// ReduceMean over the last axis, keeping it.
func (g *Graph) MeanLast(v Value) Value {
	shape := slices.Clone(v.Shape())
	shape[len(shape)-1] = 1
	return g.one("ReduceMean", "opset1", [][2]string{{"keep_dims", "true"}}, v.Type(), shape, v, g.I64s(-1))
}

// MatMul multiplies the last two dimensions, broadcasting the rest.
func (g *Graph) MatMul(a, b Value, transposeA, transposeB bool) Value {
	as, bs := slices.Clone(a.Shape()), slices.Clone(b.Shape())
	if len(as) < 2 || len(bs) < 2 {
		g.fail("ir: MatMul of %v and %v", as, bs)
		return a
	}
	if transposeA {
		as[len(as)-1], as[len(as)-2] = as[len(as)-2], as[len(as)-1]
	}
	if transposeB {
		bs[len(bs)-1], bs[len(bs)-2] = bs[len(bs)-2], bs[len(bs)-1]
	}
	k1, k2 := as[len(as)-1], bs[len(bs)-2]
	if k1 >= 0 && k2 >= 0 && k1 != k2 {
		g.fail("ir: MatMul inner dimensions %v and %v", as, bs)
	}
	batch, err := broadcast(as[:len(as)-2], bs[:len(bs)-2])
	if err != nil {
		g.fail("ir: MatMul: %w", err)
	}
	shape := append(batch, as[len(as)-2], bs[len(bs)-1])
	return g.one("MatMul", "opset1", [][2]string{
		{"transpose_a", strconv.FormatBool(transposeA)}, {"transpose_b", strconv.FormatBool(transposeB)},
	}, a.Type(), shape, a, b)
}

// Reshape with special_zero: a 0 keeps the input's dimension, a -1 is
// inferred.
func (g *Graph) Reshape(v Value, target ...int64) Value {
	in := v.Shape()
	shape := slices.Clone(target)
	known, total, infer := int64(1), int64(1), -1
	dynamic := false
	for _, d := range in {
		if d < 0 {
			dynamic = true
		}
		total *= d
	}
	for i, d := range shape {
		switch {
		case d == 0:
			shape[i] = in[i]
		case d == -1:
			infer = i
			continue
		}
		if shape[i] < 0 {
			dynamic = true
		} else {
			known *= shape[i]
		}
	}
	if infer >= 0 {
		if dynamic {
			shape[infer] = -1
		} else {
			shape[infer] = total / known
		}
	}
	return g.one("Reshape", "opset1", [][2]string{{"special_zero", "true"}}, v.Type(), shape, v, g.I64s(target...))
}

// Transpose permutes dimensions.
func (g *Graph) Transpose(v Value, perm ...int64) Value {
	in := v.Shape()
	shape := make([]int64, len(perm))
	for i, p := range perm {
		shape[i] = in[p]
	}
	return g.one("Transpose", "opset1", nil, v.Type(), shape, v, g.I64s(perm...))
}

// Slice takes [start, stop) of one axis.
func (g *Graph) Slice(v Value, axis, start, stop int64) Value {
	shape := slices.Clone(v.Shape())
	if axis < 0 {
		axis += int64(len(shape))
	}
	shape[axis] = stop - start
	return g.one("Slice", "opset8", nil, v.Type(), shape,
		v, g.I64s(start), g.I64s(stop), g.I64s(1), g.I64s(axis))
}

// Gather takes one index of an axis, dropping the axis.
func (g *Graph) Gather(v Value, axis, index int64) Value {
	in := v.Shape()
	if axis < 0 {
		axis += int64(len(in))
	}
	shape := append(slices.Clone(in[:axis]), in[axis+1:]...)
	return g.one("Gather", "opset8", [][2]string{{"batch_dims", "0"}}, v.Type(), shape,
		v, g.I64Scalar(index), g.I64Scalar(axis))
}

// Concat joins values along axis.
func (g *Graph) Concat(axis int64, vs ...Value) Value {
	shape := slices.Clone(vs[0].Shape())
	if axis < 0 {
		axis += int64(len(shape))
	}
	for _, v := range vs[1:] {
		if shape[axis] < 0 || v.Shape()[axis] < 0 {
			shape[axis] = -1
		} else {
			shape[axis] += v.Shape()[axis]
		}
	}
	return g.one("Concat", "opset1", [][2]string{{"axis", strconv.FormatInt(axis, 10)}}, vs[0].Type(), shape, vs...)
}

// Write writes the graph's XML.
func (g *Graph) Write(w io.Writer) error {
	if g.Err != nil {
		return g.Err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<?xml version=\"1.0\"?>\n<net name=%q version=\"11\">\n<layers>\n", g.Name)
	writePort := func(id int, p port, withNames bool) {
		fmt.Fprintf(&b, "<port id=\"%d\" precision=\"%s\"", id, p.t.precision())
		if withNames && p.names != "" {
			fmt.Fprintf(&b, " names=%q", p.names)
		}
		b.WriteString(">")
		for _, d := range p.shape {
			fmt.Fprintf(&b, "<dim>%d</dim>", d)
		}
		b.WriteString("</port>\n")
	}
	for _, n := range g.nodes {
		fmt.Fprintf(&b, "<layer id=\"%d\" name=%q type=%q version=%q>\n", n.id, n.name, n.kind, n.version)
		if len(n.attrs) > 0 {
			b.WriteString("<data")
			for _, a := range n.attrs {
				fmt.Fprintf(&b, " %s=%q", a[0], a[1])
			}
			b.WriteString("/>\n")
		}
		if len(n.inputs) > 0 {
			b.WriteString("<input>\n")
			for i, in := range n.inputs {
				writePort(i, in.n.outputs[in.port], false)
			}
			b.WriteString("</input>\n")
		}
		if len(n.outputs) > 0 {
			b.WriteString("<output>\n")
			for i, out := range n.outputs {
				writePort(len(n.inputs)+i, out, true)
			}
			b.WriteString("</output>\n")
		}
		b.WriteString("</layer>\n")
	}
	b.WriteString("</layers>\n<edges>\n")
	for _, n := range g.nodes {
		for i, in := range n.inputs {
			fmt.Fprintf(&b, "<edge from-layer=\"%d\" from-port=\"%d\" to-layer=\"%d\" to-port=\"%d\"/>\n",
				in.n.id, len(in.n.inputs)+in.port, n.id, i)
		}
	}
	b.WriteString("</edges>\n</net>\n")
	_, err := io.WriteString(w, b.String())
	return err
}
