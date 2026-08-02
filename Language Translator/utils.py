"""
utils.py
---------
Utility / helper functions for the AI Language Translator app.

Contains:
    - Language metadata (name <-> ISO code mappings)
    - Text statistics (characters, words, reading time)
    - Translation history persistence (JSON file based)
    - Export helpers (TXT / PDF)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "translation_history.json")

os.makedirs(HISTORY_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Supported languages (name -> ISO-639-1 code)
# 50+ languages, compatible with both MarianMT naming and M2M100 codes.
# --------------------------------------------------------------------------
LANGUAGES: Dict[str, str] = {
    "Auto Detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Polish": "pl",
    "Ukrainian": "uk",
    "Romanian": "ro",
    "Greek": "el",
    "Hebrew": "he",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Czech": "cs",
    "Slovak": "sk",
    "Hungarian": "hu",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Serbian": "sr",
    "Slovenian": "sl",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Estonian": "et",
    "Persian (Farsi)": "fa",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Nepali": "ne",
    "Sinhala": "si",
    "Burmese": "my",
    "Khmer": "km",
    "Amharic": "am",
    "Somali": "so",
    "Afrikaans": "af",
    "Zulu": "zu",
    "Xhosa": "xh",
    "Icelandic": "is",
    "Catalan": "ca",
    "Basque": "eu",
    "Galician": "gl",
}

LANGUAGE_NAMES: Dict[str, str] = {v: k for k, v in LANGUAGES.items() if v != "auto"}


def get_language_names() -> List[str]:
    """Return the list of supported language names for dropdowns."""
    return list(LANGUAGES.keys())


def code_from_name(name: str) -> str:
    """Convert a display language name to its ISO code."""
    return LANGUAGES.get(name, "en")


def name_from_code(code: str) -> str:
    """Convert an ISO code back into a display language name."""
    return LANGUAGE_NAMES.get(code, code.upper())


# --------------------------------------------------------------------------
# Text statistics
# --------------------------------------------------------------------------
def count_characters(text: str) -> int:
    return len(text)


def count_words(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())


def estimate_reading_time(text: str, wpm: int = 200) -> str:
    """Estimate reading time in seconds/minutes based on average reading speed."""
    words = count_words(text)
    if words == 0:
        return "0 sec"
    minutes = words / wpm
    seconds = int(minutes * 60)
    if seconds < 60:
        return f"{seconds} sec"
    return f"{minutes:.1f} min"


def format_processing_time(seconds: float) -> str:
    """Format elapsed processing time nicely."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


# --------------------------------------------------------------------------
# Translation history persistence
# --------------------------------------------------------------------------
def _load_raw_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw_history(history: List[Dict[str, Any]]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # Fail silently — history is a non-critical feature


def load_history() -> List[Dict[str, Any]]:
    """Load translation history, most recent first."""
    history = _load_raw_history()
    return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)


def add_history_entry(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translated_text: str,
    model_used: str,
) -> None:
    """Append a new translation entry to history."""
    history = _load_raw_history()
    entry = {
        "id": int(time.time() * 1000),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "translated_text": translated_text,
        "model_used": model_used,
        "favorite": False,
    }
    history.append(entry)
    # Keep only the most recent 100 entries to avoid unbounded growth
    history = history[-100:]
    _save_raw_history(history)


def toggle_favorite(entry_id: int) -> None:
    history = _load_raw_history()
    for entry in history:
        if entry.get("id") == entry_id:
            entry["favorite"] = not entry.get("favorite", False)
    _save_raw_history(history)


def clear_history() -> None:
    _save_raw_history([])


def search_history(query: str) -> List[Dict[str, Any]]:
    query = query.lower().strip()
    if not query:
        return load_history()
    return [
        e
        for e in load_history()
        if query in e.get("source_text", "").lower()
        or query in e.get("translated_text", "").lower()
    ]


def get_favorites() -> List[Dict[str, Any]]:
    return [e for e in load_history() if e.get("favorite")]


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------
def build_txt_bytes(source_text: str, translated_text: str, src_lang: str, tgt_lang: str) -> bytes:
    """Build a downloadable TXT file with the translation result."""
    content = (
        "AI LANGUAGE TRANSLATOR - TRANSLATION RESULT\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "=" * 50 + "\n\n"
        f"Source Language: {src_lang}\n"
        f"Original Text:\n{source_text}\n\n"
        f"Target Language: {tgt_lang}\n"
        f"Translated Text:\n{translated_text}\n"
    )
    return content.encode("utf-8")


def build_pdf_bytes(source_text: str, translated_text: str, src_lang: str, tgt_lang: str) -> Optional[bytes]:
    """Build a downloadable PDF file with the translation result using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "AI Language Translator - Result", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Source Language: {src_lang}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, source_text.encode("latin-1", "replace").decode("latin-1"))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Target Language: {tgt_lang}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, translated_text.encode("latin-1", "replace").decode("latin-1"))

    return bytes(pdf.output(dest="S"))
