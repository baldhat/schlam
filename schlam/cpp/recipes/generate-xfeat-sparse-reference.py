#!/usr/bin/env python3
"""Generate sparse-extraction test fixtures from the pinned upstream XFeat."""

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
FILES = {
    "modules/model.py": "d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964",
    "modules/xfeat.py": "385ccd31d095b0d4176b04e982088b85321b11ade4324f83b097ee6524f2a6e7",
    "modules/interpolator.py": "d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da",
    "weights/xfeat.pt": "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b",
}


def main():
    source = ROOT / "build/xfeat-upstream"
    for name, expected in FILES.items():
        path = source / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/verlab/accelerated_features/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {path}")

    import numpy as np
    import torch

    sys.path.insert(0, str(source))
    from modules.xfeat import XFeat

    torch.set_num_threads(1)
    network = XFeat(weights=str(source / "weights/xfeat.pt"), top_k=40, detection_threshold=0.01)
    network.dev = torch.device("cpu")
    network.net.cpu().eval()
    output = ROOT / "models/xfeat/reference-sparse.txt"
    with output.open("w") as stream:
        stream.write("2\n")
        for height, width in [(96, 128), (97, 131)]:
            y, x = np.indices((height, width))
            image = ((13*y + 7*x + (3*x*y) % 251 + ((x//12 + y//10) % 2)*80) % 256).astype(np.uint8)
            tensor = torch.from_numpy(image.astype(np.float32) / 255)[None, None]
            result = network.detectAndCompute(tensor)[0]
            points = result["keypoints"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            descriptors = result["descriptors"].cpu().numpy()
            stream.write(f"{height} {width} {len(points)}\n")
            for point, score, descriptor in zip(points, scores, descriptors):
                stream.write(" ".join(f"{value:.9g}" for value in [*point, score, *descriptor]) + "\n")
    metadata = {
        "revision": REVISION, "source_sha256": FILES,
        "torch_version": torch.__version__, "top_k": 40, "detection_threshold": 0.01,
        "reference_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
