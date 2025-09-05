import argparse
import pandas as pd
from collections import defaultdict

import evaluate  # HF evaluate for WER/CER


def safe_lower(s):
    return s if not isinstance(s, str) else s.strip()


def compute_metrics(df):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Overall
    refs = df["ref"].fillna("").map(safe_lower).tolist()
    hyps = df["pred"].fillna("").map(safe_lower).tolist()
    overall = {
        "wer": wer_metric.compute(references=refs, predictions=hyps),
        "cer": cer_metric.compute(references=refs, predictions=hyps),
        "num_utts": len(df),
    }

    # Per language
    per_lang = []
    for lang, g in df.groupby(df["lang"].fillna("").map(str)):
        refs = g["ref"].fillna("").map(safe_lower).tolist()
        hyps = g["pred"].fillna("").map(safe_lower).tolist()
        per_lang.append({
            "lang": lang or "-",
            "wer": wer_metric.compute(references=refs, predictions=hyps),
            "cer": cer_metric.compute(references=refs, predictions=hyps),
            "num_utts": len(g),
        })

    return overall, per_lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="results/predictions.csv from infer_whisper.py")
    ap.add_argument("--out-csv", default="results/results.csv", help="Where to write summary results")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    if "ref" not in df.columns or "pred" not in df.columns:
        raise ValueError("predictions CSV must have columns: audio_path,ref,pred,lang,...")

    overall, per_lang = compute_metrics(df)

    # Write summary CSV with header row + per-lang rows
    rows = []
    rows.append({"scope": "overall", "lang": "-", "wer": overall["wer"], "cer": overall["cer"], "num_utts": overall["num_utts"]})
    rows.extend({"scope": "per-lang", **r} for r in per_lang)

    out_df = pd.DataFrame(rows, columns=["scope", "lang", "wer", "cer", "num_utts"])
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved summary to {args.out_csv}")


if __name__ == "__main__":
    main()
