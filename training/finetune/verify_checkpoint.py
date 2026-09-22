"""One-off: verify the locally-saved fine-tuned checkpoint round-trips through
laya.load() correctly (upload it to a Modal container and try loading it there,
since torch can't be installed locally right now -- disk quota).

Usage: modal run verify_checkpoint.py
"""
from pathlib import Path

import modal

CKPT_DIR = str(Path.home() / "projects" / "gut-check" / "training" / "finetune" / "local_data" / "causal_claim_v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.48.0", "safetensors>=0.4.0", "huggingface_hub>=0.20.0", "numpy", "laya")
    .add_local_dir(CKPT_DIR, remote_path="/root/ckpt")
)

app = modal.App("gut-check-verify-checkpoint", image=image)


@app.function(gpu="T4", timeout=120)
def verify():
    import laya

    agent = laya.load("/root/ckpt")
    result = agent.system_one(
        {"claim": "All tests pass.", "context": "Bash exit code 1: 2 failed, 8 passed"},
        {"decision": {
            "type": "choice",
            "instructions": "Does the observation confirm the claim?",
            "criteria": {
                "VERIFIED": "the observation confirms the claim is true",
                "CONTRADICTED": "the observation shows the claim is false",
                "UNVERIFIED": "insufficient evidence",
                "SKIP": "not a checkable claim",
            },
        }},
    )
    return result


@app.local_entrypoint()
def main():
    r = verify.remote()
    print("loaded and ran inference successfully:")
    print(r)
