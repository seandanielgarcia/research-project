#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Dict, List
from dotenv import load_dotenv

from tqdm import tqdm
from openai import OpenAI

from utils import (
    load_cluster_file,
    get_cluster_summaries,
)

load_dotenv()


def build_clio_style_prompt(
    answers: List[str],
    contrastive_answers: List[str],
) -> str:
    joined_answers = "\n\n".join(answers)
    joined_contrastive = "\n\n".join(contrastive_answers)

    return f"""
You are tasked with generating a precise label and description for a cluster of LLM-related Reddit posts and comments.

Your goal is to summarize the dominant behavior, intent, or theme represented by the posts in this cluster and clearly distinguish it from nearby but different clusters of LLM-related discussion.

Summarize all the statements into a clear, precise, two-sentence description written in the past tense. The description must be specific to this cluster and must not apply to the contrastive examples.

After writing the description, generate a short name for the cluster.

Formatting requirements:
- The description must be exactly two sentences.
- The name must be at most ten words long.
- The name should be concrete, specific, and representative of the majority of posts.
- Avoid vague labels such as “AI discussion” or “LLM usage.”
- Assume neither good nor bad faith.
- Do not avoid naming socially harmful, sensitive, or controversial behaviors if central to the cluster.
- Do not include any text outside the tags.

Output format:

<summary>
[Insert your two-sentence summary here]
</summary>

<name>
[Insert your short cluster name here]
</name>

Below are Reddit posts and comments that belong to this cluster:

<answers>
{joined_answers}
</answers>

For context, below are Reddit posts and comments from nearby clusters that are NOT part of this cluster:

<contrastive_answers>
{joined_contrastive}
</contrastive_answers>
""".strip()


def label_cluster_clio_style(
    in_examples: List[str],
    out_examples: List[str],
    client: OpenAI,
    model_name: str,
) -> Dict[str, str]:
    prompt = build_clio_style_prompt(in_examples, out_examples)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    if not resp.choices or not resp.choices[0].message.content:
        return {"summary": "(empty)", "name": "(empty)"}

    text = resp.choices[0].message.content

    def extract(tag: str) -> str:
        start = text.find(f"<{tag}>")
        end = text.find(f"</{tag}>")
        if start == -1 or end == -1:
            return "(missing)"
        return text[start + len(tag) + 2 : end].strip()

    return {
        "summary": extract("summary"),
        "name": extract("name"),
    }


def label_clusters(
    json_path: str,
    csv_path: str,
    client: OpenAI,
    model_name: str,
    sample_per_cluster: int,
    sleep_s: float,
) -> Dict[str, Dict[str, str]]:
    clusters = load_cluster_file(json_path)
    cluster_names = list(clusters.keys())
    outputs = {}

    for cluster_name in tqdm(cluster_names, desc="Labeling clusters", ncols=80):
        in_examples = get_cluster_summaries(
            cluster_name,
            json_path,
            csv_path,
        )

        if not in_examples:
            outputs[cluster_name] = {
                "summary": "(no content)",
                "name": "(no content)",
            }
            continue

        if sample_per_cluster > 0:
            in_examples = in_examples[:sample_per_cluster]

        contrastive = []
        for other in cluster_names:
            if other == cluster_name:
                continue
            contrastive.extend(
                get_cluster_summaries(other, json_path, csv_path)
            )
            if len(contrastive) >= len(in_examples):
                break

        contrastive = contrastive[: len(in_examples)]

        result = label_cluster_clio_style(
            in_examples=in_examples,
            out_examples=contrastive,
            client=client,
            model_name=model_name,
        )

        outputs[cluster_name] = result

        if sleep_s > 0:
            time.sleep(sleep_s)

    return outputs


def analyst_review(labels: Dict[str, Dict[str, str]], client, model_name: str) -> str:
    combined = "\n".join(
        f"{k}: {v['name']}" for k, v in labels.items()
    )

    prompt = (
        "You are reviewing cluster labels generated from LLM-related Reddit data.\n\n"
        "Tasks:\n"
        "- Identify obvious redundancy or near-duplicate labels\n"
        "- Comment if clusters appear overly granular or overly broad\n"
        "- Suggest whether hierarchical grouping may be useful\n\n"
        "Do NOT rename clusters.\n"
        "Return 3–5 concise analytical sentences.\n\n"
        f"{combined}"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    if not resp.choices or not resp.choices[0].message.content:
        return "(no review output)"

    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser(description="Clio-style cluster labeling pipeline")
    ap.add_argument("--clusters", required=True, help="Path to clusters JSON")
    ap.add_argument("--csv", required=True, help="CSV with post data")
    ap.add_argument("--out", default="cluster_labels.json", help="Output JSON file")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model name")
    ap.add_argument("--sample", type=int, default=12, help="Examples per cluster")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between calls")
    ap.add_argument("--api-key", help="OpenAI API key")

    args = ap.parse_args()

    api_key = (
        args.api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
    )
    if not api_key:
        raise SystemExit("Set --api-key or export OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    labels = label_clusters(
        json_path=args.clusters,
        csv_path=args.csv,
        client=client,
        model_name=args.model,
        sample_per_cluster=args.sample,
        sleep_s=args.sleep,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    review = analyst_review(labels, client, args.model)
    with open(args.out.replace(".json", "_review.txt"), "w", encoding="utf-8") as f:
        f.write(review)


if __name__ == "__main__":
    main()
