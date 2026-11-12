#!/usr/bin/env python3
"""Prepare SQuALITY v1.3 multi-question prompts for shared-prefix KV experiments.

The output contains the *actual* token IDs, not just synthetic trace lengths.
No timestamps are invented: arrival processes should be supplied separately.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

PREFIX = "Read the following document and answer the question.\n\nDocument:\n"
SUFFIX = "\n\nQuestion:\n"
BLOCK_SIZE = 16


def shared_prefix_length(sequences):
    if not sequences:
        return 0
    first = sequences[0]
    bound = min(map(len, sequences))
    for pos in range(bound):
        if any(seq[pos] != first[pos] for seq in sequences[1:]):
            return pos
    return bound


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="SQuALITY v1.3 data directory (train/dev/test.jsonl)")
    p.add_argument("--tokenizer", required=True, help="Exact tokenizer used by the inference model")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-doc-tokens", type=int, default=0)
    p.add_argument("--max-doc-tokens", type=int, default=1000000)
    p.add_argument("--min-questions", type=int, default=5)
    p.add_argument("--min-reference-output-tokens", type=int, default=0,
                   help="Minimum median reference length per question (0 disables filter)")
    p.add_argument("--fixed-output-tokens", type=int, default=0,
                   help="0 = reference-summary median; positive = fixed synthetic generation budget")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="If positive, require input_length + output_length <= this context limit")
    p.add_argument("--omit-token-ids", action="store_true",
                   help="Smaller file, but direct token-ID replay will not be possible")
    a = p.parse_args()
    if not (0 <= a.min_doc_tokens <= a.max_doc_tokens):
        p.error("Require 0 <= min-doc-tokens <= max-doc-tokens")
    if a.min_questions < 1 or a.fixed_output_tokens < 0 or a.max_model_len < 0:
        p.error("Invalid negative parameter or min-questions < 1")

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    req_path, docs_path = out / "squality_requests.jsonl", out / "squality_docs.csv"
    doc_stats = []
    doc_total = doc_kept = requests_kept = 0

    with req_path.open("w", encoding="utf-8") as dst:
        for split in ("train", "dev", "test"):
            source = Path(a.data_dir) / f"{split}.jsonl"
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open(encoding="utf-8") as src:
                for no, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc_total += 1
                    doc_id = f"{split}-{no:04d}"
                    document = row["document"]
                    doc_tokens = len(encode(document))
                    base = PREFIX + document + SUFFIX
                    distinct, seen = [], set()
                    for question in row["questions"]:
                        text = question.get("question_text", "").strip()
                        if text and text.casefold() not in seen:
                            seen.add(text.casefold())
                            distinct.append(question)

                    eligible = []
                    if a.min_doc_tokens <= doc_tokens <= a.max_doc_tokens:
                        for qi, question in enumerate(distinct, start=1):
                            text = question["question_text"].strip()
                            refs = [x["response_text"] for x in question.get("responses", [])
                                    if isinstance(x, dict) and x.get("response_text")]
                            ref_lengths = [len(encode(x)) for x in refs]
                            ref_median = round(statistics.median(ref_lengths)) if ref_lengths else None
                            if a.min_reference_output_tokens and (ref_median is None or ref_median < a.min_reference_output_tokens):
                                continue
                            output_length = (a.fixed_output_tokens or ref_median or 256)
                            output_length = max(1, output_length)
                            prompt = base + text + "\nAnswer:"
                            token_ids = encode(prompt)
                            if a.max_model_len and len(token_ids) + output_length > a.max_model_len:
                                continue
                            eligible.append(dict(
                                request_id=f"{doc_id}-q{qi}", doc_id=doc_id,
                                split=split, question_id=qi, question=text,
                                prompt=prompt, prompt_token_ids=token_ids,
                                input_length=len(token_ids), document_tokens=doc_tokens,
                                output_length=output_length,
                                output_length_source="fixed" if a.fixed_output_tokens else
                                ("reference_median" if ref_median is not None else "fallback_256"),
                                reference_median_tokens=ref_median,
                            ))

                    selected = len(eligible) >= a.min_questions
                    common = shared_prefix_length([r["prompt_token_ids"] for r in eligible]) if selected else 0
                    doc_stats.append(dict(
                        doc_id=doc_id, split=split, doc_tokens=doc_tokens,
                        questions=len(distinct), eligible_questions=len(eligible),
                        shared_prefix_tokens=common,
                        shared_prefix_blocks=common // BLOCK_SIZE,
                        selected=int(selected),
                    ))
                    if not selected:
                        continue
                    doc_kept += 1
                    for request in eligible:
                        request["shared_prefix_tokens"] = common
                        request["shared_prefix_blocks"] = common // BLOCK_SIZE
                        if a.omit_token_ids:
                            request.pop("prompt_token_ids")
                        dst.write(json.dumps(request, ensure_ascii=False) + "\n")
                        requests_kept += 1

    with docs_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(doc_stats[0]) if doc_stats else
                                ["doc_id", "split", "doc_tokens", "questions", "eligible_questions",
                                 "shared_prefix_tokens", "shared_prefix_blocks", "selected"])
        writer.writeheader()
        writer.writerows(doc_stats)

    selected_docs = [r for r in doc_stats if r["selected"]]
    print(f"All: {doc_total} documents; selected: {doc_kept} documents / {requests_kept} requests")
    if selected_docs:
        print(f"Median actual shared prefix: {statistics.median(r['shared_prefix_tokens'] for r in selected_docs):.0f} tokens")
    print(f"Requests: {req_path}\nDocuments: {docs_path}")
    print("No request-arrival timestamps generated. Output lengths are reference-based or fixed budgets, not measured model outputs.")
    print("Important: existing synthetic make_prompt() trace replay must be replaced with direct prompt_token_ids replay.")


if __name__ == "__main__":
    main()
