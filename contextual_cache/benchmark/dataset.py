"""
Dataset loader for benchmarking.

Downloads a subset of Natural Questions from HuggingFace and generates
paraphrase variants for testing semantic matching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "dataset_cache"


@dataclass
class BenchmarkQuery:
    """A single query for benchmarking."""
    id: str
    question: str
    answer: str
    is_paraphrase: bool = False
    original_id: str = ""


def _paraphrase(q: str, seed: int) -> str:
    """
    Simple deterministic paraphrasing via rule-based transforms.
    Not ML-based — keeps it reproducible without external services.
    """
    rng = random.Random(seed)
    transforms = [
        # Swap "What is" / "What are" → "Explain" / "Describe"
        (r"^What is ", "Explain "),
        (r"^What are ", "Describe "),
        (r"^Who is ", "Tell me about "),
        (r"^Who was ", "Describe "),
        (r"^When did ", "In what year did "),
        (r"^Where is ", "What is the location of "),
        (r"^How many ", "What is the number of "),
        (r"^How does ", "Explain how "),
        (r"^Why ", "For what reason "),
    ]

    text = q
    for pattern, replacement in transforms:
        if re.match(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            break
    else:
        # Generic: add "Can you tell me" prefix or append "?"
        prefixes = [
            "Can you explain ",
            "I'd like to know ",
            "Tell me ",
            "Please describe ",
        ]
        text = rng.choice(prefixes) + text[0].lower() + text[1:]

    # Remove trailing ? and re-add
    text = text.rstrip("?").strip() + "?"
    return text


def load_benchmark_dataset(
    num_questions: int = 500,
    paraphrase_ratio: float = 0.3,
    seed: int = 42,
) -> List[BenchmarkQuery]:
    """
    Load NQ questions and generate paraphrased variants.

    Returns a deterministic list of BenchmarkQuery objects.
    Caches the processed dataset locally so subsequent runs are instant.
    """
    cache_file = CACHE_DIR / f"nq_{num_questions}_p{int(paraphrase_ratio*100)}_s{seed}.json"

    if cache_file.exists():
        logger.info("Loading cached benchmark dataset from %s", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkQuery(**d) for d in data]

    logger.info("Downloading NQ dataset from HuggingFace (first run only)…")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        )

    # Load the simplified NQ dataset
    ds = load_dataset("LLukas22/nq-simplified", split="train")

    # Take a deterministic subset
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:num_questions]

    queries: List[BenchmarkQuery] = []
    for idx in selected:
        row = ds[idx]
        question = row.get("question", "").strip()
        # Extract short answer
        answers = row.get("answers", [])
        if isinstance(answers, dict):
            answer_text = answers.get("text", [""])[0] if "text" in answers else ""
        elif isinstance(answers, list) and answers:
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("text", "")
            else:
                answer_text = str(answers[0])
        else:
            answer_text = str(answers) if answers else ""

        if not question or not answer_text:
            continue

        qid = hashlib.md5(question.encode()).hexdigest()[:12]
        queries.append(BenchmarkQuery(
            id=qid,
            question=question,
            answer=answer_text,
        ))

    # Generate paraphrases
    num_paraphrases = int(len(queries) * paraphrase_ratio)
    paraphrase_indices = rng.sample(range(len(queries)), min(num_paraphrases, len(queries)))

    paraphrased: List[BenchmarkQuery] = []
    for pi in paraphrase_indices:
        orig = queries[pi]
        para_q = _paraphrase(orig.question, seed + pi)
        para_id = hashlib.md5(para_q.encode()).hexdigest()[:12]
        paraphrased.append(BenchmarkQuery(
            id=para_id,
            question=para_q,
            answer=orig.answer,
            is_paraphrase=True,
            original_id=orig.id,
        ))

    # Interleave: originals first, then paraphrases mixed in
    all_queries = queries + paraphrased
    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_queries)

    # Cache for next time
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([{
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "is_paraphrase": q.is_paraphrase,
            "original_id": q.original_id,
        } for q in all_queries], f, indent=2)
    logger.info("Cached %d benchmark queries to %s", len(all_queries), cache_file)

    return all_queries
