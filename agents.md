# eg2npu

EmbeddingGemma 2's text encoder on an Intel NPU (or the CPU or GPU) through
OpenVINO, served over the part of llama-server's API an embedding client uses.
It reads the GGUF llama-server loads and must land where llama-server's
vectors land: khoj's index was embedded by llama-server, and a query vector
from anywhere else has to be comparable with it. Text only; images and audio
stay with llama-server.

## Layout

- `gguf/`: maps a GGUF and reads its metadata and tensors (F32, F16, BF16,
  Q8_0), dequantized the way ggml does.
- `tokenizer/`: the Gemma 4 tokenizer, a port of llama.cpp's
  `LLAMA_VOCAB_PRE_TYPE_GEMMA4` path. Token for token what `llama-tokenize`
  gives; `testdata/llama.json` pins 26 inputs.
- `ir/`: writes OpenVINO IR v11 (XML plus a weights file) with shapes on every
  port. Constants go through a content addressed `Pool`, so the graphs for all
  sequence lengths share one weights file.
- `eg2/`: `Build` writes llama.cpp's `src/models/gemma-embedding2.cpp` as IR;
  `Model` writes the IR into the cache, compiles it per length, embeds, pools
  and normalizes.
- `server/`: `POST /v1/embeddings` (and `/embeddings`), `GET /v1/models`,
  `/props`, `/health`, in llama-server's shapes.
- `cmd/eg2npu`: the server. `cmd/eg2cmp`: compares with a llama-server or
  saved vectors and times each input. `cmd/eg2tok`: token ids per file, for
  diffing against `llama-tokenize --ids --no-escape`. `cmd/ggufdump`,
  `cmd/ovdevices`: diagnostics.
- `hf/`: downloads the GGUF from Hugging Face for `-hf owner/repo:QUANT`, as
  llama-server's `-hf` does. The file is checked against the size and sha256
  the repository lists before it is renamed into place, and a file already
  downloaded is used without the network, so a start does not depend on
  Hugging Face. `mmproj-Q8_0.gguf` ends like the model and is skipped.
  `FetchFiles` fetches a given list of paths from a repository the same way
  (bolo uses it for its OpenVINO export); it lives here until `hf` moves to
  a module of its own.
- The OpenVINO bindings are `github.com/daaku/openvino` (`~/workspace/openvino`,
  split out of whispy, shared with it). They can read, compile and run a model,
  not build or reshape one, which is why the graph is written as IR text.

## Matching llama.cpp

Everything that decides where a vector lands follows llama.cpp, not the
Hugging Face model card:

- BOS and EOS both: the GGUF sets `add_eos_token`, and llama.cpp forces
  `add_bos` for Gemma 4. Special tokens written in the text are parsed, as
  llama-server does for embeddings.
- The sliding window is the GGUF's 1024, symmetric: a token sees those within
  512 on either side (`is_masked_swa`, `LLAMA_SWA_TYPE_SYMMETRIC`). The HF
  config says 512; it only matters past 513 tokens.
- NEOX rope over the whole head (256 on sliding layers with base 1e4, 512 on
  the global ones with base 1e6). The angles are carried in float32 and
  stepped by `base^(-2/d)`, as `ggml_rope_cache_init` does.
- Attention is not scaled by 1/sqrt(d) (`f_attention_scale = 1`); q and k are
  RMS normed with weights, v without.
- GELU is the tanh approximation (`ggml_gelu`).
- Mean pooling over every token including BOS and EOS, then L2 normalization
  (`common_embd_normalize`, norm 2). The output projection is applied per
  token before pooling; it is linear, so the order does not matter.

Measured against llama-server (Q8_0, `--pooling mean`) on 24 inputs of 10 to
1948 tokens: mean cosine 0.99993 on the CPU (fp32 compute) and the GPU, 0.99992
on the NPU (fp16), worst 0.99982. llama-server on two machines agreed with itself to
1.000000. Through khojd, 10 of 10 queries returned the same top 5 with the
same order.

## The NPU

- Static shapes only, so there is one graph per length in `DefaultLengths`
  and an input is padded to the first that holds it; the mask hides padding
  as keys and pooling skips it. CPU and GPU use one dynamic graph.
- Token embeddings are looked up on the host (a 262144 x 512 table, Q8_0
  rows dequantized on demand from the mapped GGUF). Masks and rope tables are
  computed on the host in float32 and passed in, so nothing position
  dependent is computed in the NPU's float16.
- fp16 is enough: no residual scaling was needed (`ResidualScale` stays 1; it
  exists and is tested as an escape hatch).
- **Lengths that equal a feature width fault the device.** With driver 1.38
  and OpenVINO 2026.4 on Panther Lake, a graph whose sequence length was 512
  (the hidden size) or 2048 (the FFN width) compiled, then failed its first
  inference with `zeFenceHostSynchronize: ZE_RESULT_ERROR_UNKNOWN`; 504, 576,
  1024, 1040 and 2064 ran. `DefaultLengths` avoids them, and `SelfCheck` runs
  every compiled length at startup so a bad one stops the server instead of a
  query.
- Attention runs in blocks of query rows past `MaxScores` (768 x 768), and a
  sliding window block reads only the keys its window reaches. Whole
  attention at 1024 faulted the same way before the blocking.
- The per layer input projection is computed per layer: whole it is
  `[1, L, 12288]`, and the 2048 graph faulted with it.
- The NPU computes in fp16 whatever the weights are stored as, so `q8`
  weights only make it compile 2.5x slower; `-weights auto` is q8 on the CPU
  alone (a third faster there) and f16 on the NPU and GPU.
- Every length is its own compiled model with its own copy of the weights,
  about 285 MB each on the NPU (the blobs in `CACHE_DIR` are that size, and
  the server maps them). The nine default lengths come to 2.9 GB resident,
  against 0.48 GB for one length and 1.1 GB for `-lengths 64,384,2064`.
  The GPU (0.75 GB) and CPU (0.58 GB) servers hold one copy.
- First compile of all lengths takes about 80 s; `CACHE_DIR` keeps the blobs,
  and a later start takes under 2 s.

## The GPU

- Static lengths too (`-shapes auto` is dynamic on the CPU only). With one
  dynamic graph the Arc B390 took 27 ms a query whenever the length changed
  from the last one, against 4.6 ms static.
- f16 weights (`-weights auto` is q8 on the CPU only). With q8 the GPU ran
  slower and drifted further from llama-server (worst cosine 0.99964 against
  0.99988).
- In boe's SSH sandbox `/etc/OpenCL/vendors` is not bound, and the GPU plugin
  finds no device without it: `OCL_ICD_VENDORS` pointing at a directory with
  an `intel.icd` naming `/usr/lib/intel-opencl/libigdrcl.so` brings it back.

## Measured on boe

Core Ultra X7 358H (Panther Lake, Arc B390, NPU driver 1.38, OpenVINO
2026.4.1), on battery. Energy is RAPL package energy above an idle baseline
taken just before; the queries are 10 to 21 tokens, the passages about 450.

| | query p50 | queries/s | J/query | passages/s | J/passage | 1 query/s: J/query, p50 |
| --- | --- | --- | --- | --- | --- | --- |
| eg2npu NPU | 5.6 ms | 158 | 0.07 | 21.6 | 0.68 | 0.04, 17.5 ms |
| eg2npu GPU | 4.6 ms | 199 | 0.10 | 61 | 0.59 | 0.09, 16.7 ms |
| eg2npu CPU | 10 ms | 82 | 0.34 | 4.4 | 6.8 | 0.27, 26 ms |
| llama.cpp CPU | 27 ms | 36 | 0.59 | 1.0 | 19 | 0.57, 66 ms |

The energy comes from `/sys/class/powercap/intel-rapl:0/energy_uj` (root only
unless made readable) and the NPU's own
`/sys/class/accel/accel0/device/npu_busy_time_us`. The battery's `power_now`
lags by tens of seconds, so it is no use for anything shorter than a minute.

## Packaging

- `PKGBUILD` builds `eg2npu` and installs `eg2npu.service`, a system unit
  with `DynamicUser=yes`. The NPU (`/dev/accel/accel0`) and the GPU belong to
  `render`, hence `SupplementaryGroups=render`. The GGUF goes in
  `StateDirectory` (`$STATE_DIRECTORY/models`), the IR and the compiled models
  in `CacheDirectory`, and the NPU driver's own cache under
  `XDG_CACHE_HOME`, pointed into the cache directory because a dynamic user's
  home is `/`.
- `eg2npu.socket` listens on `127.0.0.1:8741` and starts the service on the
  first connection; `github.com/daaku/lands` picks up the socket
  (`LISTEN_FDS`), else `PORT`, `ADDR`, then `-listen`, and shuts down on
  SIGTERM. Connections queue in the socket's backlog while the model loads.
- The unit runs `-lengths 64`: khojd's queries fit, and every length costs
  about 285 MB on the NPU. It listens on 8741, llama-server's port in khoj's
  setup, so it cannot run beside a llama-server there.

## Build and test

- `go build ./cmd/eg2npu`. cgo links `libopenvino_c`; build here and copy the
  binary to a machine with the same OpenVINO.
- `go test ./...`. The model tests need the Q8_0 GGUF at
  `~/.cache/eg2/embeddinggemma-2-Q8_0.gguf` (or `$EG2_GGUF`) and skip
  without it. `EG2_DEVICE=NPU go test ./eg2/` runs the parity test on the NPU.
- Regenerating `testdata/llama.json` needs llama.cpp's `llama-tokenize` and a
  `llama-server --embeddings --pooling mean` on the same GGUF.
