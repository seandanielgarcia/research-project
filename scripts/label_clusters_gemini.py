#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Dict, List

from tqdm import tqdm
import google.generativeai as genai
from utils import (
    load_cluster_file,
    get_cluster_summaries,
)

def make_prompt(cluster_name: str, examples: List[str], max_examples: int) -> str:
    trimmed = [s for s in examples if s]
    # If max_examples > 0, cap the number of examples; otherwise include all
    if max_examples > 0:
        trimmed = trimmed[:max_examples]

    joined_examples = "\n\n".join(f"Example {i+1}: {ex}" for i, ex in enumerate(trimmed))
    return (
        "You are labeling clusters of Reddit posts about AI behavior. "
        "Your task is to produce a **distinct, descriptive label (4–8 words)** that captures what makes this cluster "
        "unique compared to other possible clusters.\n\n"
        "Guidelines:\n"
        "-The label should be general enough to unite all posts in the cluster not just one of the exaples"
        "- Avoid generic phrases like 'AI discussion' or 'general opinions about AI'.\n"
        "- Emphasize what specifically unites these posts — e.g. topic, attitude, focus, or controversy.\n"
        "- Avoid punctuation except basic hyphens.\n"
        "- Do not label any cluster until you have seen all posts in all clusters"
        "- Return only the label.\n\n"
        
        f"Cluster name (internal): {cluster_name}\n\n"
        f"Examples:\n{joined_examples}"
    )

def call_gemini(prompt: str, model) -> str:
    try:
        resp = model.generate_content(prompt)
        if not hasattr(resp, "text") or not resp.text:
            return "(empty response)"
        return resp.text.strip().splitlines()[0].strip("\" ")
    except Exception as e:
        return f"(error: {e.__class__.__name__})"

def label_clusters(json_path: str, csv_path: str, model, sample_per_cluster: int, sleep_s: float) -> Dict[str, str]:
    clusters = load_cluster_file(json_path)
    labels = {}

    for cluster_name in tqdm(clusters.keys(), desc="Labeling clusters", ncols=80):
        examples = get_cluster_summaries(cluster_name, json_path, csv_path)
        if not examples:
            labels[cluster_name] = "(no content)"
            continue

        prompt = make_prompt(cluster_name, examples, sample_per_cluster)
        label = call_gemini(prompt, model)
        labels[cluster_name] = label

        if sleep_s > 0:
            time.sleep(sleep_s)

    return labels

def evaluate_cluster_quality(labels: Dict[str, str], model) -> str:
    """Ask Gemini to evaluate whether clusters seem too granular or overlapping."""
    combined_labels = "\n".join(f"{k}: {v}" for k, v in labels.items())
    prompt = (
        "You are an AI clustering evaluator.\n"
        "Given a list of cluster labels from a dataset of Reddit posts about AI, "
        "analyze whether the clusters are too similar or too numerous.\n\n"
        "Tasks:\n"
        "- Identify if some clusters overlap in theme or phrasing.\n"
        "- Comment if the total number of clusters seems excessive (too granular) or too few (too broad).\n"
        "- Suggest roughly how many clusters might be optimal.\n"
        "- Be concise but analytical.\n\n"
        "Cluster labels:\n"
        f"{combined_labels}\n\n"
        "Return your analysis in 3–5 sentences."
    )
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, "text") and resp.text else "(no evaluation output)"
    except Exception as e:
        return f"(error during evaluation: {e.__class__.__name__})"

def main():
    ap = argparse.ArgumentParser(description="Label clusters using Gemini with distinctness and evaluation")
    ap.add_argument("--clusters", required=True, help="Path to clusters JSON")
    ap.add_argument("--csv", required=True, help="CSV with post data")
    ap.add_argument("--out", default="cluster_labels.json", help="Output JSON file")
    ap.add_argument("--model", default="gemini-1.5-pro", help="Gemini model name")
    ap.add_argument("--sample", type=int, default=-1, help="Examples per cluster (-1 to include ALL posts)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between API calls")
    ap.add_argument("--api-key", help="Google API key (fallback: GOOGLE_API_KEY env)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("gemini_key")
    if not api_key:
        raise SystemExit("Set --api-key or export GOOGLE_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    print("🔹 Generating cluster labels...")
    labels = label_clusters(
        json_path=args.clusters,
        csv_path=args.csv,
        model=model,
        sample_per_cluster=args.sample,
        sleep_s=args.sleep,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(labels)} cluster labels → {args.out}")

    print("\n🔹 Evaluating cluster distinctiveness...")
    evaluation = evaluate_cluster_quality(labels, model)
    print("\n=== Cluster Evaluation ===")
    print(evaluation)
    print("==========================")

    eval_out = args.out.replace(".json", "_evaluation.txt")
    with open(eval_out, "w", encoding="utf-8") as f:
        f.write(evaluation)
    print(f"📄 Evaluation saved → {eval_out}")

if __name__ == "__main__":
    main()
