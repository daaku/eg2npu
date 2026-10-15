importpath=github.com/daaku/eg2npu
pkgname=$(basename "$importpath")
pkgver=$(git rev-list --count HEAD)
pkgrel=1
pkgdesc='EmbeddingGemma 2 text embeddings on an Intel NPU, served over the llama-server API'
arch=('x86_64')
url="https://$importpath"
license=('MIT')
depends=(
  'glibc'                     # libc
  'libgcc'                    # libgcc_s
  'libstdc++'                 # libstdc++
  'openvino'                  # libopenvino_c, runs the model
  'openvino-intel-npu-plugin' # the NPU, the service's device
)
makedepends=('go')
optdepends=(
  'openvino-intel-gpu-plugin: -device GPU'
)

build() {
  cd ..

  # makepkg's LDFLAGS do not reach the cgo link, so the hardening flags are
  # passed here explicitly.
  export CGO_ENABLED=1
  export CGO_LDFLAGS='-Wl,-z,relro,-z,now'
  go build -trimpath -o eg2npu ./cmd/eg2npu
}

package() {
  cd ..
  install -Dm755 eg2npu "$pkgdir/usr/bin/eg2npu"
  install -Dm644 eg2npu.service "$pkgdir/usr/lib/systemd/system/eg2npu.service"
  install -Dm644 license "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
