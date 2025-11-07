#!/usr/bin/env python3
"""Upload a local merged model to Hugging Face Hub."""

import argparse
import os
from pathlib import Path

from huggingface_hub import upload_folder, get_repo_url


def upload_model(local_path: str, repo_id: str, private: bool = False):
    """Upload a model directory to Hugging Face Hub.
    
    Args:
        local_path: Local path to the model directory
        repo_id: Hugging Face repo ID (e.g., username/model-name)
        private: Whether to make the repo private
    """
    local_path = Path(local_path).expanduser().resolve()
    
    if not local_path.exists():
        print(f"❌ Error: Model path does not exist: {local_path}")
        return False
    
    if not local_path.is_dir():
        print(f"❌ Error: Path is not a directory: {local_path}")
        return False
    
    # Check for model files
    has_model = any(local_path.glob("*.safetensors")) or any(local_path.glob("*.bin"))
    has_config = (local_path / "config.json").exists()
    
    if not has_model or not has_config:
        print(f"❌ Error: Directory doesn't contain model files (safetensors/bin + config.json)")
        return False
    
    print(f"📦 Uploading model from: {local_path}")
    print(f"🚀 Destination: {repo_id}")
    print(f"🔒 Private: {private}")
    
    try:
        # Upload the folder
        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            repo_type="model",
            private=private,
        )
        
        print(f"\n✅ Model successfully uploaded!")
        print(f"📍 Repository URL: {repo_url}")
        print(f"\n🎉 You can now use your model with:")
        print(f"   from transformers import AutoModel")
        print(f"   model = AutoModel.from_pretrained('{repo_id}')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\n💡 Make sure you:")
        print(f"   1. Have huggingface_hub installed: pip install huggingface_hub")
        print(f"   2. Are logged in: huggingface-cli login")
        print(f"   3. Have write access to {repo_id}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a merged model to Hugging Face Hub"
    )
    parser.add_argument(
        "model_path",
        help="Local path to the model directory to upload",
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repo ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    
    args = parser.parse_args()
    
    success = upload_model(args.model_path, args.repo_id, private=args.private)
    exit(0 if success else 1)
