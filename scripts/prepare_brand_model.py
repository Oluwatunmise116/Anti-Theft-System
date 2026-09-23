#!/usr/bin/env python3
"""
Fetch the whole-vehicle brand model into models/car_brands_beit/.

The running app never downloads weights; this is the explicit offline step.
It pins the exact revision that was evaluated on this gate's crops
(attributes/brand_classifier.py MODEL_REVISION), and fetches only the files
inference needs: the repo also carries optimizer.pt and scheduler.pt, which
are training state.

    python scripts/prepare_brand_model.py
    python scripts/prepare_brand_model.py --dest models/car_brands_beit

After fetching, restart the app (or save any attr_* setting) so the
attribute pipeline is rebuilt with `attr_brand_backend: auto` picking it up.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from attributes.brand_classifier import MODEL_ID, MODEL_REVISION  # noqa: E402

FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", default=os.path.join(ROOT, "models", "car_brands_beit"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             allow_patterns=FILES, local_dir=args.dest)
    print(f"{MODEL_ID}@{MODEL_REVISION[:7]} -> {path}")

    # Prove it loads offline before declaring success.
    from attributes.brand_classifier import VehicleBrandClassifier
    model = VehicleBrandClassifier(path)
    if not model.is_ready():
        print(f"Downloaded, but failed to load: {model.status()['load_error']}")
        return 1
    print(f"Loaded: {len(model.id2label)} make+model classes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
