# XFeat ONNX backbone

`xfeat.onnx` contains the official pretrained XFeat backbone, exported at ONNX
opset 17. It embeds its weights and needs no Python installation at runtime.
Source: [verlab/accelerated_features](https://github.com/verlab/accelerated_features),
revision `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`. The upstream Apache 2.0
license is included in `LICENSE`. `xfeat.json` records source/checkpoint/model
checksums, export versions and the PyTorch/ONNX Runtime parity results.

## C++ usage

```cpp
#include "schlam/feature_detection/xfeat/XFeatModel.h"

XFeatModel model("models/xfeat/xfeat.onnx"); // Load once.
auto output = model.run(grayscaleImage);   // Reuse for every frame.
```

The input is a two-dimensional `CV_8UC1` grayscale image. Both dimensions must
be at least 32 and divisible by 32. The loader converts pixels to float32 in
`[0,1]`. The model includes grayscale instance normalization. The model's input
name is `image`, with shape `[1,1,H,W]`; H and W are dynamic.

The returned `cv::Mat` objects own their memory and have float32 NCHW layout:

| Output | Shape | Meaning |
| --- | --- | --- |
| `features` | `[1,64,H/8,W/8]` | Dense, unnormalized descriptors |
| `keypointLogits` | `[1,65,H/8,W/8]` | Keypoint logits, including dustbin |
| `reliability` | `[1,1,H/8,W/8]` | Reliability probabilities |

The loader verifies model metadata, names, tensor types and shapes. It uses CPU
inference with a configurable thread count. ONNX Runtime details are hidden
behind the `XFeatModel` interface.

To exercise the actual C++ loader after building:

```sh
source build/Release/generators/conanrun.sh
build/Release/xfeat_model
build/Release/xfeat_model models/xfeat/xfeat.onnx path/to/grayscale-image.png
```

The first command uses a synthetic 96x128 image; the optional image must meet the
dimension requirements above. Pass an explicit model path when deploying outside
the source checkout. The standalone executable's default points at this checkout.

This is the backbone/model-loading layer. Sparse extraction still needs the
upstream heatmap decoding, nonmaximum suppression, reliability ranking, descriptor
sampling and L2 normalization. An extractor should resize images to dimensions
divisible by 32 and map selected coordinates back to the original image. Existing
SLAM configs continue to use ORB.

## Reproduce the model

From `schlam/cpp`, create an isolated export environment:

```sh
python3 -m venv build/xfeat-export-venv
build/xfeat-export-venv/bin/python -m pip install -r recipes/xfeat-export-requirements.txt
build/xfeat-export-venv/bin/python recipes/export-xfeat.py
```

The exporter downloads the pinned architecture, checkpoint and license into
`build/xfeat-upstream` and verifies all three SHA-256 checksums. It uses
`torch.load(weights_only=True)`. Cached verified files support offline re-export.
`--source-dir` and `--output` override the cache and model paths.

For dynamic-shape ONNX export, the wrapper replaces `Tensor.unfold` with an
equivalent reshape/permute. It compares all three outputs against the unmodified
upstream PyTorch implementation at 32x32, 64x96 and 96x128 with absolute/relative
tolerance `2e-4`. It also saves full upstream reference tensors for the deterministic
32x32 image `image[y,x] = (13*y + 7*x) % 256` in `reference-32x32.txt`. The C++ tests
check their inference outputs against those reference tensors.
