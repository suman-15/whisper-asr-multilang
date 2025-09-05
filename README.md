# Whisper Large-v3 Inference & Evaluation (Chinese, Pashto, Urdu)

This repository provides a **reproducible** pipeline to run inference with Hugging Face's
[`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) on **Chinese, Pashto, and Urdu**
audio, and to **evaluate** results into a CSV. It also includes a **GitHub Pages** site that renders the
latest `results/results.csv` so your project **displays results** publicly.

> ✅ Designed to be **forkable** and easy to run on any system (Windows/Linux/macOS).  
> ⚠️ Full-model evaluation is **GPU-intensive**. CI is configured to **only publish results you push**; it does **not**
run full inference on GitHub Actions.

---

## Quickstart

1. **Clone or fork** this repo, then create and activate a Conda env (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate whisper-eval
   ```

   Alternatively with `pip`:
   ```bash
   python -m venv .venv
   . .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare your data** (any format is OK as long as you can produce a manifest CSV):
   - Create a manifest CSV with columns:
     - `audio_path` — absolute or repo-relative path to a WAV/MP3/FLAC file
     - `text` — (optional) reference transcription
     - `lang` — language code among: `zh`, `ps`, `ur` (Chinese/Pashto/Urdu)
   - Example: `data/manifest.csv`
     ```csv
     audio_path,text,lang
     data/zh/sample1.wav,你好世界,zh
     data/ps/sample2.wav,,ps
     data/ur/sample3.wav,یہ ایک مثال ہے,ur
     ```

3. **Run inference** (writes `results/predictions.csv`):
   ```bash
   python src/infer_whisper.py --input-manifest data/manifest.csv --model openai/whisper-large-v3 --device auto
   ```

4. **Evaluate** (writes `results/results.csv` and prints a per-language table):
   ```bash
   python scripts/eval_and_save.py --predictions results/predictions.csv --out-csv results/results.csv
   ```

5. **Publish results to a web page (GitHub Pages)**:
   - Commit and push to `main`. The included workflow publishes `docs/` as a site.
   - Make sure **GitHub Pages** is enabled for your repo with **Source: GitHub Actions**.
   - Place your latest CSV at `results/results.csv` and push — the site will read and display it.

---

## Project Structure

```
.
├─ src/
│  └─ infer_whisper.py         # CLI for Whisper inference -> results/predictions.csv
├─ scripts/
│  └─ eval_and_save.py         # Computes WER/CER etc. -> results/results.csv
├─ notebooks/
│  └─ whisper_cup_asr.ipynb    # Your original notebook (copied here)
├─ results/
│  ├─ predictions.csv          # (generated) per-utterance predictions
│  └─ results.csv              # (generated) aggregated metrics + per-lang
├─ docs/
│  ├─ index.html               # GitHub Pages site (renders results CSV as a table)
│  └─ app.js
├─ .github/workflows/pages.yml  # Publishes docs/ to GitHub Pages
├─ environment.yml
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## Notes & Tips

- **Hardware**: `whisper-large-v3` is heavy — a GPU with ≥16GB VRAM is ideal. For CPU-only or low-VRAM tests,
  you can try `--model openai/whisper-small` just to validate the pipeline.
- **Languages**: We pass `--task transcribe` and set `--language` per row when known. If `lang` is missing, the model
  can auto-detect, but for best comparability keep the field filled.
- **Attention mask warning**: The provided script constructs the batch with the processor so the
  `attention_mask` is correctly set even when pad/eos tokens coincide in Whisper.
- **Reproducibility**: Pin exact versions in `requirements.txt`/`environment.yml` if you need bit-for-bit runs.
- **Privacy**: Don't commit raw audio unless you have rights to publish. Results CSVs are fine.

---

## License

This project is released under the **MIT License** (see `LICENSE`).

---

## Citation

If you use Whisper, please also cite the original paper and model card.
