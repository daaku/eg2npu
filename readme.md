# eg2npu

EmbeddingGemma 2 text embeddings on an Intel NPU, served over llama-server's
embeddings API, from the same GGUF llama-server loads. The vectors match
llama-server's (cosine ≥ 0.9998), so it can embed queries against an index
that llama-server built.

```sh
eg2npu -lengths 64
khojd -model http://127.0.0.1:8741 serve
```

The first start downloads the Q8_0 GGUF from
[unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF)
(295 MB, checked against its sha256) and compiles it for the NPU. Later starts
use both from disk. `-model` takes a GGUF you already have instead.

On Arch, `makepkg -si` installs it with a system service:

```sh
systemctl enable --now eg2npu
```

The service runs as a dynamic user in the `render` group, keeps the model in
`/var/lib/eg2npu` and the compiled graphs in `/var/cache/eg2npu`, and serves
queries up to 64 tokens on `127.0.0.1:8741`.

It handles text only. `khojd index` (and `run`) embed pictures and sound too,
so they stay on llama-server. `khojd serve` only embeds queries, so it can use
either one. The name in `/v1/models` is `embeddinggemma2`, llama-server's
`--alias` in khoj's setup, so khojd sees the same model and keeps its index.

## Flags

| Flag | Default | |
| --- | --- | --- |
| `-hf` | `unsloth/embeddinggemma-2-GGUF:Q8_0` | Repository and quantization to download, as llama-server's `-hf` |
| `-models` | `$STATE_DIRECTORY/models`, or `models` in the cache | Where `-hf` downloads go |
| `-model` | | A local EmbeddingGemma 2 GGUF instead of `-hf` (Q8_0 is what khoj uses) |
| `-device` | `NPU` | `NPU`, `GPU` or `CPU` |
| `-listen` | `127.0.0.1:8741` | |
| `-alias` | `embeddinggemma2` | The name in `/v1/models` |
| `-cache` | `$CACHE_DIRECTORY` or `~/.cache/eg2npu` | The IR written from the GGUF, and compiled blobs |
| `-shapes` | `auto` | `static` (one graph per length), `dynamic`, or `auto`: static on the NPU (which needs it) and the GPU (6x faster with it), dynamic on the CPU |
| `-lengths` | `32,64,128,256,384,640,1024,1536,2064` | The static lengths. Avoid 512 and 2048 (see agents.md). Each one costs about 285 MB on the NPU |
| `-max-tokens` | `2048` | The longest input, like llama-server's `-ub 2048` |
| `-weights` | `auto` | `f16`, or `q8` (Q8_0's own int8). `auto` uses q8 on the CPU and f16 on the NPU and GPU |

The first start writes the IR (about 8 s) and compiles every length for the
device (about 80 s on the NPU). Later starts take under 2 s. At startup it
embeds one text at every length and checks the results agree. A length the
device miscompiles stops the server there, not on a query.

## API

`POST /v1/embeddings` (and `/embeddings`) takes `input` as a string, a list of
strings, a list of token ids, or a list of those. `encoding_format` is
`float` or `base64`. Each vector is mean pooled and L2 normalized, as from
`llama-server --embeddings --pooling mean`. Inputs get BOS and EOS, and
special tokens written in the text are parsed, as llama-server does. Image or
audio items are refused with a 400. An input over `-max-tokens` is a 500
saying so, as llama-server answers.

`GET /v1/models`, `/props` (`n_ctx` is `-max-tokens`, one slot) and `/health`
answer as llama-server does.
