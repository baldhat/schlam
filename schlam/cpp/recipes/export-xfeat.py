#!/usr/bin/env python3
"""Export and verify the official pretrained XFeat backbone (CPU, ONNX opset 17)."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import urllib.request

REVISION = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
FILES = {
    "modules/model.py": "d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964",
    "weights/xfeat.pt": "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
ROOT = Path(__file__).resolve().parents[1]


def verified_source(directory):
    for name, expected in FILES.items():
        destination = directory / name
        if not destination.exists():
            url = f"https://raw.githubusercontent.com/verlab/accelerated_features/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {url}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "build/xfeat-upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "models/xfeat/xfeat.onnx")
    args = parser.parse_args()
    verified_source(args.source_dir)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec = importlib.util.spec_from_file_location("xfeat_upstream_model", args.source_dir / "modules/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.XFeatModel().cpu().eval()
    checkpoint = torch.load(args.source_dir / "weights/xfeat.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint)
    # Tensor.unfold cannot be exported with dynamic spatial dimensions by the
    # legacy exporter. This equivalent reshape/permute preserves channel order.
    class ExportableXFeat(module.XFeatModel):
        def _unfold2d(self, image, ws=2):
            batch, channels, height, width = image.shape
            blocks = image.reshape(batch, channels, height // ws, ws, width // ws, ws)
            return blocks.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * ws * ws, height // ws, width // ws)

    export_network = ExportableXFeat().cpu().eval()
    export_network.load_state_dict(checkpoint)
    output_names = ["features", "keypoint_logits", "reliability"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            export_network, torch.rand(1, 1, 64, 96), str(args.output),
            input_names=["image"], output_names=output_names,
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                **{name: {2: "height_div8", 3: "width_div8"} for name in output_names},
            },
            opset_version=17, dynamo=False,
        )
    model = onnx.load(str(args.output))
    # The exporter leaves symbolic output batch dimensions after InstanceNorm,
    # although the input batch is fixed at one. Make that contract explicit.
    for output in model.graph.output:
        output.type.tensor_type.shape.dim[0].dim_value = 1
    metadata = {
        "schlam.model": "xfeat-backbone",
        "schlam.xfeat_revision": REVISION,
        "schlam.checkpoint_sha256": FILES["weights/xfeat.pt"],
        "schlam.input": "float32 NCHW grayscale [0,1]; batch=1; H,W>=32 and divisible by 32",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    parity = []
    for height, width in ((32, 32), (64, 96), (96, 128)):
        image = torch.rand(1, 1, height, width)
        with torch.inference_mode():
            expected = network(image)
        actual = session.run(output_names, {"image": image.numpy()})
        errors = {}
        for name, reference, result in zip(output_names, expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), atol=2e-4, rtol=2e-4)
            errors[name] = float(np.max(np.abs(result - reference.numpy())))
        parity.append({"input_shape": list(image.shape), "maximum_absolute_errors": errors})
    # Full-tensor golden data comes from the unmodified PyTorch implementation.
    # The C++ loader tests use the same deterministic uint8 image, then /255.
    rows, columns = np.indices((32, 32))
    pixels = ((13 * rows + 7 * columns) % 256).astype(np.uint8)
    with torch.inference_mode():
        reference_outputs = network(torch.from_numpy(pixels).float()[None, None] / 255.0)
    reference_path = args.output.parent / "reference-32x32.txt"
    np.savetxt(reference_path, np.concatenate([value.numpy().ravel() for value in reference_outputs]), fmt="%.9g")
    manifest = {
        "upstream": "https://github.com/verlab/accelerated_features",
        "revision": REVISION,
        "source_sha256": FILES,
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "opset": 17,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "parity": parity,
        "reference": {
            "file": reference_path.name,
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "input": "uint8 image[y,x] = (13*y + 7*x) % 256; shape 32x32; /255 before inference",
            "layout": "flattened NCHW features, keypoint_logits, reliability, in that order",
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(args.source_dir / "LICENSE", args.output.parent / "LICENSE")
    print(json.dumps(manifest, indent=2))
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
