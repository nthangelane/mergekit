# Uploading GA-Generated Models to Hugging Face Hub

## Quick Start

### 1. Install Required Packages
```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face
```bash
huggingface-cli login
# You'll be prompted for your Hugging Face API token
# Get it from: https://huggingface.co/settings/tokens
```

### 3. Upload Your Best Model

#### Option A: Using the Upload Script (Easiest)
```bash
python upload_to_hf.py workspace/tiny_cpu_ga_10gen/final_model username/model-name
```

#### Option B: Using evolve_ga.py directly (Automatic)
When running GA, add the `--hf-model-id` flag:
```bash
python -m mergekit.scripts.evolve_ga workspace/tiny_cpu_ga_experiment.yml \
  --storage-path workspace/tiny_cpu_ga_10gen \
  --max-fevals 120 \
  --save-final-model \
  --hf-model-id username/model-name
```

#### Option C: Manual Upload with Python
```python
from huggingface_hub import upload_folder

upload_folder(
    repo_id="username/model-name",
    folder_path="workspace/tiny_cpu_ga_10gen/final_model",
    repo_type="model",
)
```

## Important Notes

### API Token Setup
- **Local file**: `~/.huggingface/token`
- **Environment variable**: `HF_TOKEN=your_token_here`
- **Get token**: https://huggingface.co/settings/tokens

### Repository Naming
- Format: `username/model-name`
- The repository will be created automatically if it doesn't exist
- Make sure your account has write permissions

### Model Files Included
When you upload a merged model, it includes:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` & `tokenizer.config.json` - Tokenizer files
- `mergekit_config.yml` - The merge configuration used
- `README.md` - Auto-generated model card

### Usage After Upload
Once uploaded, users can load your model with:
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("username/model-name")
tokenizer = AutoTokenizer.from_pretrained("username/model-name")
```

## Example Commands

### Upload with private flag
```bash
python upload_to_hf.py workspace/tiny_cpu_ga_10gen/final_model username/private-model --private
```

### Upload from different storage path
```bash
python upload_to_hf.py /path/to/any/model/directory username/model-name
```

## Troubleshooting

### "403 Forbidden" Error
- Check your API token is valid
- Ensure you're logged in: `huggingface-cli whoami`
- Check repo permissions at https://huggingface.co/settings/tokens

### "Repository not found"
- The repo will be created automatically
- Make sure you have write access to your account

### Connection Issues
- Check internet connection
- Try `huggingface-cli repo-list` to test connectivity

## What Gets Uploaded

From `final_model/`:
```
.
├── model.safetensors       (182MB) - The merged weights
├── config.json             - Model architecture config
├── tokenizer.json          - Tokenizer vocabulary
├── tokenizer_config.json   - Tokenizer settings
├── special_tokens_map.json - Special token mappings
├── mergekit_config.yml     - Merge recipe (for reproducibility)
└── README.md               - Model card
```

## Next Steps

1. Upload the model
2. Visit https://huggingface.co/username/model-name
3. Add a detailed description and tags
4. Enable model card features
5. Share with the community!
