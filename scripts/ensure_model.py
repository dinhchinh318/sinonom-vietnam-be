import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/var/data/models/finetuning-phase2"))
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # ví dụ "username/finetuning-phase2"
HF_REVISION = os.environ.get("HF_REVISION")  # optional

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Marker file để biết đã tải xong
    done = MODEL_DIR / ".downloaded"
    if done.exists() and any(MODEL_DIR.iterdir()):
        print(f"[model] OK: exists at {MODEL_DIR}")
        return

    if not HF_REPO_ID:
        raise RuntimeError("Missing HF_REPO_ID. Either mount/copy model or set HF_REPO_ID to download.")

    print(f"[model] Downloading from HF: {HF_REPO_ID} -> {MODEL_DIR}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    done.write_text("ok")
    print("[model] Download complete.")

if __name__ == "__main__":
    main()
