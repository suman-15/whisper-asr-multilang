import os
import csv
import argparse
from pathlib import Path
from typing import Optional, List

import torch
from tqdm import tqdm
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "audio_path": r.get("audio_path", "").strip(),
                "text": r.get("text", "").strip() if "text" in r else "",
                "lang": r.get("lang", "").strip()
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Whisper inference over a manifest CSV.")
    parser.add_argument("--input-manifest", type=str, required=True, help="CSV with columns: audio_path,text(optional),lang")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="pipeline batch size")
    parser.add_argument("--fp16", action="store_true", help="use fp16 when cuda is available")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--output", type=str, default="results/predictions.csv")
    args = parser.parse_args()

    manifest = read_manifest(Path(args.input_manifest))
    if not manifest:
        raise ValueError("Empty manifest or unreadable CSV.")

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = 0  # assume cuda:0

    # Load model & processor explicitly so attention_mask is handled via processor padding
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if (args.fp16 and torch.cuda.is_available()) else None,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        batch_size=args.batch_size,
        torch_dtype=torch.float16 if (args.fp16 and torch.cuda.is_available()) else None,
    )

    # Ensure output dir exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "ref", "pred", "lang", "task", "model"])

        for row in tqdm(manifest, desc="Transcribing"):
            audio_path = row["audio_path"]
            ref = row.get("text", "")
            lang = row.get("lang", "") or None  # allow None for auto-detect

            # Set language per item if available (improves consistency)
            generate_kwargs = {"task": args.task}
            if lang:
                # whisper expects language name or code; codes like 'zh', 'ps', 'ur' generally work
                generate_kwargs["language"] = lang

            # Call pipeline (it will build attention_mask via processor)
            try:
                result = asr(audio_path, generate_kwargs=generate_kwargs)
                pred = result["text"]
            except Exception as e:
                pred = f"[ERROR] {e}"

            writer.writerow([audio_path, ref, pred, lang or "", args.task, args.model])

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
