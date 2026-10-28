"""Streaming CPU inference for TextScan's experimental writing-pattern analysis.

Runs in a dedicated spawned process. DistilGPT-2 is loaded and warmed up as soon as
the process starts, so the first scan does not pay the model-load cost. Each completed
sentence is sent through a result queue, and in-flight jobs can be cancelled.
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import importlib
import math
import queue
import re
import signal
import statistics
import sys
from functools import lru_cache
from typing import Any, Callable

MODEL_ID = "distilbert/distilgpt2"
MAX_SENTENCES = 120
MAX_TOKENS_PER_SENTENCE = 1024
THREADS = int(os.environ.get("TEXTSCAN_THREADS", min(4, os.cpu_count() or 1)))
_LOG_CAP = math.log(1_000_000)
_WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class _Cancelled(Exception):
    pass


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -value))))


def _classify_sentence(
    mean_nll: float | None,
    words: int,
    average_words: float,
    length_stddev: float,
    length_variation: float,
) -> tuple[str, str, float]:
    """Return a visualization cue, label, and uncalibrated predictability signal."""
    signal_value = _sigmoid(4.5 - mean_nll) if mean_nll is not None else 0.0
    if signal_value >= 0.65:
        return "orange", "Higher model predictability", signal_value

    varies = (
        length_stddev > 0
        and length_variation >= 0.18
        and abs(words - average_words) >= max(1.0, 0.55 * length_stddev)
    )
    if varies:
        return "green", "Sentence length varies from this passage", signal_value
    return "neutral", "No standout pattern", signal_value


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    """Import, load, and warm up the model (worker process only)."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.set_num_threads(THREADS)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    model.eval()
    with torch.inference_mode():  # one throwaway pass initializes kernels and allocators
        model(input_ids=tokenizer("Warm up the model.", return_tensors="pt")["input_ids"])
    return torch, tokenizer, model


def analyze_text_stream(
    text: str,
    emit: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Emit start, sentence, and complete events without waiting for the whole text."""
    sentences = _split_sentences(text)
    if not sentences:
        raise ValueError("No sentences were found in the submitted text.")
    if len(sentences) > MAX_SENTENCES:
        raise ValueError(f"Please submit no more than {MAX_SENTENCES} sentences at a time.")

    word_counts = [_word_count(s) for s in sentences]
    total_words = sum(word_counts)
    average_words = statistics.fmean(word_counts)
    length_stddev = statistics.pstdev(word_counts) if len(word_counts) > 1 else 0.0
    length_variation = length_stddev / average_words if average_words else 0.0
    sample_note = (
        "Short sample: these measurements are especially unstable below 100 words."
        if total_words < 100
        else "This experimental heuristic is not calibrated and does not establish authorship."
    )

    emit({
        "type": "start",
        "model": MODEL_ID,
        "sentence_count": len(sentences),
        "word_count": total_words,
        "sample_note": sample_note,
    })

    torch, tokenizer, model = _load_model()
    nlls: list[float] = []
    scored = 0
    rounded_variation = round(length_variation, 3)

    for index, (sentence, words) in enumerate(zip(sentences, word_counts), start=1):
        if is_cancelled():
            raise _Cancelled

        ids = tokenizer(sentence, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        original_tokens = len(ids)
        if original_tokens > MAX_TOKENS_PER_SENTENCE:
            ids = ids[:MAX_TOKENS_PER_SENTENCE]
        token_count = len(ids)

        mean_nll: float | None = None
        perplexity: float | None = None
        if token_count >= 2:
            tensor = torch.tensor([ids])
            with torch.inference_mode():
                mean_nll = float(model(input_ids=tensor, labels=tensor).loss.item())
            nlls.append(mean_nll)
            scored += 1
            perplexity = round(math.exp(min(mean_nll, _LOG_CAP)), 2)

        highlight, highlight_label, signal_value = _classify_sentence(
            mean_nll, words, average_words, length_stddev, length_variation
        )
        emit({
            "type": "sentence",
            "sentence": index,
            "total_sentences": len(sentences),
            "snippet": sentence[:180],
            "snippet_truncated": len(sentence) > 180,
            "words": words,
            "tokens": token_count,
            "perplexity": perplexity,
            "predictability_signal": round(signal_value, 3),
            "length_variation": rounded_variation,
            "truncated": original_tokens > MAX_TOKENS_PER_SENTENCE,
            "highlight": highlight,
            "highlight_label": highlight_label,
        })

    if not nlls:
        raise ValueError("The text is too short to score. Add more words and try again.")

    mean_nll = statistics.fmean(nlls)
    predictability = _sigmoid(3.5 - mean_nll)
    uniformity = 1.0 / (1.0 + length_variation)
    emit({
        "type": "complete",
        "model": MODEL_ID,
        "pattern_score": round(100 * (0.70 * predictability + 0.30 * uniformity)),
        "mean_sentence_perplexity": round(math.exp(min(mean_nll, _LOG_CAP)), 2),
        "sentence_length_variation": rounded_variation,
        "word_count": total_words,
        "sentence_count": len(sentences),
        "sample_note": sample_note,
        "scored_sentences": scored,
    })


def worker_main(job_queue: Any, result_queue: Any, cancel_queue: Any, ready: Any) -> None:
    """Warm the model, then process jobs and stream progress events back."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        _load_model()
        ready.set()
    except Exception as error:  # jobs will report the failure; the server stays up
        print(f"model warm-up failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    cancelled: set[str] = set()

    def drain_cancellations() -> None:
        while True:
            try:
                cancelled.add(cancel_queue.get_nowait())
            except queue.Empty:
                return

    while True:
        job = job_queue.get()
        if job is None:
            break
        request_id = job["request_id"]

        def emit(event: dict[str, Any], _rid: str = request_id) -> None:
            result_queue.put({"request_id": _rid, **event})

        def is_cancelled(_rid: str = request_id) -> bool:
            drain_cancellations()
            return _rid in cancelled

        try:
            if is_cancelled():
                raise _Cancelled
            analyze_text_stream(job["text"], emit, is_cancelled)
            ready.set()
        except _Cancelled:
            emit({"type": "error", "message": "The scan was cancelled."})
        except ValueError as error:
            emit({"type": "error", "message": str(error)})
        except Exception as error:
            emit({"type": "error", "message": f"Analysis failed ({type(error).__name__}). Please retry later."})
        finally:
            cancelled.discard(request_id)
            if len(cancelled) > 256:  # stale ids from requests that finished before their cancel arrived
                cancelled.clear()

    result_queue.put(None)
