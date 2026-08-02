"""
translator.py
--------------
Core AI translation engine.

Strategy:
    1. Try to load a dedicated MarianMT model for the requested language
       pair (Helsinki-NLP/opus-mt-{src}-{tgt}). These models are small,
       fast, and highly accurate for the pairs they were trained on.
    2. If no MarianMT model exists for that pair, fall back to
       Facebook's M2M100 multilingual model, which supports 100 languages
       in any direction.
    3. Language auto-detection is done with `langdetect` when the user
       selects "Auto Detect".

All models are cached with `st.cache_resource` so they are downloaded /
loaded into memory only once per session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
import torch

# --------------------------------------------------------------------------
# Known MarianMT (Helsinki-NLP/opus-mt-*) language pairs.
# This is not exhaustive of every pair Helsinki-NLP publishes, but covers
# the most common high-traffic pairs. Anything outside this set
# automatically falls back to M2M100.
# --------------------------------------------------------------------------
MARIAN_SUPPORTED_PAIRS = {
    ("en", "es"), ("es", "en"),
    ("en", "fr"), ("fr", "en"),
    ("en", "de"), ("de", "en"),
    ("en", "it"), ("it", "en"),
    ("en", "pt"), ("pt", "en"),
    ("en", "nl"), ("nl", "en"),
    ("en", "ru"), ("ru", "en"),
    ("en", "zh"), ("zh", "en"),
    ("en", "ja"), ("ja", "en"),
    ("en", "ko"), ("ko", "en"),
    ("en", "ar"), ("ar", "en"),
    ("en", "hi"), ("hi", "en"),
    ("en", "tr"), ("tr", "en"),
    ("en", "vi"), ("vi", "en"),
    ("en", "pl"), ("pl", "en"),
    ("en", "sv"), ("sv", "en"),
    ("en", "fi"), ("fi", "en"),
    ("en", "el"), ("el", "en"),
    ("en", "he"), ("he", "en"),
    ("en", "uk"), ("uk", "en"),
    ("en", "ro"), ("ro", "en"),
    ("en", "cs"), ("cs", "en"),
    ("en", "da"), ("da", "en"),
    ("en", "id"), ("id", "en"),
    ("es", "fr"), ("fr", "es"),
    ("es", "de"), ("de", "es"),
    ("de", "fr"), ("fr", "de"),
}

MARIAN_MODEL_TEMPLATE = "Helsinki-NLP/opus-mt-{src}-{tgt}"
M2M100_MODEL_NAME = "facebook/m2m100_418M"


@dataclass
class TranslationResult:
    translated_text: str
    model_used: str
    detected_language: Optional[str]
    processing_time: float


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """Detect the ISO-639-1 language code of the input text."""
    try:
        from langdetect import detect

        code = detect(text)
        # langdetect uses 'zh-cn' for Chinese; normalize to 'zh'
        if code.startswith("zh"):
            code = "zh"
        return code
    except Exception:
        return "en"  # Safe fallback


# --------------------------------------------------------------------------
# Model loading (cached across the session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_marian_model(src: str, tgt: str):
    """Load and cache a MarianMT model + tokenizer for a specific pair."""
    from transformers import MarianMTModel, MarianTokenizer

    model_name = MARIAN_MODEL_TEMPLATE.format(src=src, tgt=tgt)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(get_device())
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_m2m100_model():
    """Load and cache the multilingual M2M100 model + tokenizer."""
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(M2M100_MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(M2M100_MODEL_NAME)
    model.to(get_device())
    model.eval()
    return tokenizer, model


def pair_supported_by_marian(src: str, tgt: str) -> bool:
    return (src, tgt) in MARIAN_SUPPORTED_PAIRS


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
def translate_with_marian(text: str, src: str, tgt: str) -> Tuple[str, str]:
    tokenizer, model = load_marian_model(src, tgt)
    device = get_device()
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generated = model.generate(**batch, max_length=512)
    result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    model_name = MARIAN_MODEL_TEMPLATE.format(src=src, tgt=tgt)
    return result, model_name


def translate_with_m2m100(text: str, src: str, tgt: str) -> Tuple[str, str]:
    tokenizer, model = load_m2m100_model()
    device = get_device()
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt),
            max_length=512,
        )
    result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return result, M2M100_MODEL_NAME


def translate(text: str, src: str, tgt: str) -> TranslationResult:
    """
    Main translation entry point.

    - Auto-detects the source language if src == 'auto'.
    - Prefers MarianMT for known pairs, falls back to M2M100 otherwise.
    - Returns a TranslationResult with timing + model metadata.
    """
    start = time.time()
    detected_language = None

    if src == "auto":
        detected_language = detect_language(text)
        src = detected_language

    if src == tgt:
        # No translation needed; just echo back
        elapsed = time.time() - start
        return TranslationResult(
            translated_text=text,
            model_used="No translation required (same language)",
            detected_language=detected_language,
            processing_time=elapsed,
        )

    try:
        if pair_supported_by_marian(src, tgt):
            translated_text, model_used = translate_with_marian(text, src, tgt)
        else:
            translated_text, model_used = translate_with_m2m100(text, src, tgt)
    except Exception:
        # If the specific MarianMT pair fails to download / doesn't exist,
        # gracefully fall back to the universal M2M100 model.
        translated_text, model_used = translate_with_m2m100(text, src, tgt)

    elapsed = time.time() - start
    return TranslationResult(
        translated_text=translated_text,
        model_used=model_used,
        detected_language=detected_language,
        processing_time=elapsed,
    )
