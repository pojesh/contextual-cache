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
    Deterministic paraphrasing via rule-based transforms.
    Produces more linguistically diverse variants to better test semantic matching.
    """
    rng = random.Random(seed)
    transforms = [
        (r"^What is ", "Define "),
        (r"^What is ", "Give me a definition of "),
        (r"^What are ", "List "),
        (r"^What are ", "What do we mean by "),
        (r"^Who is ", "Tell me about "),
        (r"^Who is ", "Can you identify "),
        (r"^Who was ", "Identify the person known as "),
        (r"^When did ", "In which year did "),
        (r"^When did ", "What year saw "),
        (r"^Where is ", "What is the location of "),
        (r"^Where is ", "Locate "),
        (r"^How many ", "What's the count of "),
        (r"^How many ", "Give me the number of "),
        (r"^How does ", "Explain the functioning of "),
        (r"^How does ", "What is the mechanism behind "),
        (r"^Why do ", "What causes "),
        (r"^Why do ", "Explain the reason for "),
        (r"^Why is ", "What explains "),
        (r"^Why is ", "For what reason is "),
        (r"^Can you ", ""),
        (r"^Could you ", ""),
        (r"^Please ", ""),
        (r"^Tell me ", "What is "),
        (r"^I want to know ", ""),
        (r"^I'd like to know ", ""),
    ]

    text = q
    applied = False
    for pattern, replacement in transforms:
        if re.match(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            applied = True
            break

    if not applied:
        prefixes = [
            "Here's a question: ",
            "I need to know: ",
            "Quick question: ",
            "Mind telling me ",
            "Curious about ",
            "Looking for info on ",
        ]
        text = rng.choice(prefixes) + text[0].lower() + text[1:]

    text = text.rstrip("?").strip()
    if not text.endswith((".", "!")):
        text += "?"

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


def load_squad_dataset(
    num_questions: int = 500,
    paraphrase_ratio: float = 0.3,
    seed: int = 42,
) -> List[BenchmarkQuery]:
    """
    Load SQuAD validation questions and generate paraphrased variants.

    Uses the same BenchmarkQuery format and paraphrase logic as NQ.
    """
    cache_file = CACHE_DIR / f"squad_{num_questions}_p{int(paraphrase_ratio*100)}_s{seed}.json"

    if cache_file.exists():
        logger.info("Loading cached SQuAD dataset from %s", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkQuery(**d) for d in data]

    logger.info("Downloading SQuAD dataset from HuggingFace (first run only)…")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        )

    ds = load_dataset("rajpurkar/squad", split="validation")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    queries: List[BenchmarkQuery] = []
    seen_questions: set = set()
    for idx in indices:
        if len(queries) >= num_questions:
            break
        row = ds[idx]
        question = row.get("question", "").strip()
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer_text = answer_texts[0].strip() if answer_texts else ""

        if not question or not answer_text or question in seen_questions:
            continue
        seen_questions.add(question)

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

    all_queries = queries + paraphrased
    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_queries)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([{
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "is_paraphrase": q.is_paraphrase,
            "original_id": q.original_id,
        } for q in all_queries], f, indent=2)
    logger.info("Cached %d SQuAD benchmark queries to %s", len(all_queries), cache_file)

    return all_queries


def load_dataset_by_name(
    name: str = "nq",
    num_questions: int = 500,
    paraphrase_ratio: float = 0.3,
    seed: int = 42,
) -> List[BenchmarkQuery]:
    """Dispatcher for loading benchmark datasets by name."""
    if name == "nq":
        return load_benchmark_dataset(
            num_questions=num_questions,
            paraphrase_ratio=paraphrase_ratio,
            seed=seed,
        )
    elif name == "squad":
        return load_squad_dataset(
            num_questions=num_questions,
            paraphrase_ratio=paraphrase_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: nq, squad")
