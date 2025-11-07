# Quick Start: GA Model Evolution & Upload

## ⚡ 5-Minute Quick Start

### 1️⃣ Run GA Experiment
```bash
cd /Users/nkululekothangelane/Documents/master_research/mergekit
python -m mergekit.scripts.evolve_ga workspace/tiny_cpu_ga_experiment.yml \
  --storage-path workspace/my_experiment \
  --max-fevals 120 \
  --no-merge-cuda \
  --num-gpus 0 \
  --save-final-model
```

### 2️⃣ Upload to Hugging Face
```bash
# First time only: authenticate
huggingface-cli login

# Upload your best model
python upload_to_hf.py workspace/my_experiment/final_model username/model-name
```

**Done!** Your model is now on Hugging Face Hub 🎉

---

## 📊 What You Get

| Item | Location | Size | Notes |
|------|----------|------|-------|
| **Best Merged Model** | `final_model/model.safetensors` | 182 MB | Ready to use |
| **Merge Config** | `final_model/mergekit_config.yml` | - | Reproducible |
| **Tokenizer** | `final_model/tokenizer.json` | 2 MB | Included |
| **Model Cache** | `transformers_cache/` | 926 MB | Reusable across runs |

---

## 🚀 Advanced Options

### With Automatic HF Upload
```bash
python -m mergekit.scripts.evolve_ga workspace/tiny_cpu_ga_experiment.yml \
  --storage-path workspace/my_experiment \
  --max-fevals 120 \
  --no-merge-cuda \
  --num-gpus 0 \
  --save-final-model \
  --hf-model-id username/model-name  # ← Auto-upload enabled!
```

### Reuse Cache from Previous Run
```bash
python -m mergekit.scripts.evolve_ga workspace/tiny_cpu_ga_experiment.yml \
  --storage-path workspace/my_experiment \
  --max-fevals 120 \
  --no-merge-cuda \
  --num-gpus 0 \
  --no-reshard  # ← Skip model resharding
```

### Multiple Runs with Different Seeds
```bash
for seed in 42 123 456; do
  python -m mergekit.scripts.evolve_ga workspace/tiny_cpu_ga_experiment.yml \
    --storage-path workspace/exp_$seed \
    --max-fevals 120 \
    --random-seed $seed \
    --no-merge-cuda --num-gpus 0 \
    --save-final-model \
    --hf-model-id username/model-$seed
done
```

---

## 📚 Using Your Uploaded Model

### Load in Python
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("username/model-name")
tokenizer = AutoTokenizer.from_pretrained("username/model-name")

# Generate text
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Share on Hugging Face
1. Visit: `https://huggingface.co/username/model-name`
2. Click **Edit model card**
3. Add description, tags, and usage examples
4. Save & share!

---

## 🐛 Troubleshooting

### "Permission denied" during upload
```bash
# Make sure you're logged in
huggingface-cli whoami

# If not, login first
huggingface-cli login
```

### "Repository not found"
- The repo will be created automatically
- Make sure your HF account has enough space

### GA runs too slow
- Use `--max-fevals 20` for a quick test (5 min)
- Increase to `--max-fevals 300+` for better results
- Use GPU if available (remove `--no-merge-cuda`)

### Models keep re-downloading
- Keep the same `--storage-path` to reuse cache
- Use `--no-reshard` on subsequent runs

---

## 📖 More Info

- **Full Guide**: See `GA_EXPERIMENT_SUMMARY.md`
- **Upload Help**: See `HF_UPLOAD_GUIDE.md`
- **Current Results**: See `workspace/tiny_cpu_ga_10gen/final_model/`

---

## 🎯 Typical Workflow

```
1. Run GA              (30 minutes)  → best model saved
2. Login to HF         (1 minute)    → authenticate
3. Upload model        (2 minutes)   → appears on HF Hub
4. Share & showcase!   (forever)     → community access
```

---

**Status**: ✅ Ready to Use
**Last Updated**: October 19, 2025
