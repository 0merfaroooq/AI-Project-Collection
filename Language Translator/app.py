"""
app.py
------
AI Language Translator — a premium SaaS-style Streamlit application
powered by local Hugging Face Transformer models (MarianMT / M2M100).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from styles import inject_custom_css
from translator import get_device, translate
from utils import (
    LANGUAGES,
    add_history_entry,
    build_pdf_bytes,
    build_txt_bytes,
    clear_history,
    code_from_name,
    count_characters,
    count_words,
    estimate_reading_time,
    format_processing_time,
    get_language_names,
    load_history,
    name_from_code,
    search_history,
    toggle_favorite,
)

# ==========================================================================
# Page configuration
# ==========================================================================
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css(st)

# ==========================================================================
# Session state initialization
# ==========================================================================
DEFAULTS = {
    "source_text": "",
    "translated_text": "",
    "last_model_used": "—",
    "last_processing_time": 0.0,
    "last_detected": None,
    "source_lang_name": "English",
    "target_lang_name": "Spanish",
    "history_search": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def do_swap_languages() -> None:
    """Swap source and target languages (and their text, if both are set)."""
    if st.session_state.source_lang_name != "Auto Detect":
        st.session_state.source_lang_name, st.session_state.target_lang_name = (
            st.session_state.target_lang_name,
            st.session_state.source_lang_name,
        )
        st.session_state.source_text, st.session_state.translated_text = (
            st.session_state.translated_text,
            st.session_state.source_text,
        )


def do_clear() -> None:
    st.session_state.source_text = ""
    st.session_state.translated_text = ""
    st.session_state.last_model_used = "—"
    st.session_state.last_processing_time = 0.0
    st.session_state.last_detected = None


# ==========================================================================
# Sidebar — Recent Translations / History
# ==========================================================================
with st.sidebar:
    st.markdown("### 🕘 Translation History")

    search_query = st.text_input(
        "Search history", value=st.session_state.history_search, placeholder="Search past translations..."
    )
    st.session_state.history_search = search_query

    tab_recent, tab_favs = st.tabs(["Recent", "⭐ Favorites"])

    entries = search_history(search_query) if search_query else load_history()

    with tab_recent:
        if not entries:
            st.caption("No translations yet. Start translating to build your history!")
        for entry in entries[:15]:
            star = "⭐" if entry.get("favorite") else "☆"
            with st.container():
                st.markdown(
                    f"""
                    <div class="history-item">
                        <span class="history-lang">{entry['source_lang']} → {entry['target_lang']}</span><br/>
                        <b>{entry['source_text'][:60]}{'...' if len(entry['source_text']) > 60 else ''}</b><br/>
                        <span style="color:#94A3B8;">{entry['translated_text'][:60]}{'...' if len(entry['translated_text']) > 60 else ''}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cols = st.columns([1, 1])
                if cols[0].button(star, key=f"fav_{entry['id']}"):
                    toggle_favorite(entry["id"])
                    st.rerun()
                if cols[1].button("Use", key=f"use_{entry['id']}"):
                    st.session_state.source_text = entry["source_text"]
                    st.session_state.translated_text = entry["translated_text"]
                    st.rerun()

    with tab_favs:
        favs = [e for e in entries if e.get("favorite")]
        if not favs:
            st.caption("No favorites yet. Tap ☆ on any translation to save it here.")
        for entry in favs:
            st.markdown(
                f"""
                <div class="history-item">
                    <span class="history-lang">{entry['source_lang']} → {entry['target_lang']}</span><br/>
                    <b>{entry['source_text'][:60]}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All History", use_container_width=True):
        clear_history()
        st.toast("History cleared", icon="🗑️")
        st.rerun()

    st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)
    st.caption(f"⚙️ Compute device: **{get_device().upper()}**")


# ==========================================================================
# Hero section
# ==========================================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">POWERED BY HUGGING FACE TRANSFORMERS</div>
        <div class="hero-title">🌍 AI Language Translator</div>
        <div class="hero-subtitle">
            Translate text instantly across 50+ languages using state-of-the-art
            Transformer models — running entirely locally, no external APIs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Metric cards
# ==========================================================================
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">50+</div>
            <div class="metric-label">Languages Supported</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">MarianMT / M2M100</div>
            <div class="metric-label">Translation Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.last_processing_time and format_processing_time(st.session_state.last_processing_time) or '—'}</div>
            <div class="metric-label">Inference Speed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)

# ==========================================================================
# Main translator: Input | Swap | Output
# ==========================================================================
col_input, col_swap, col_output = st.columns([5, 1, 5])

lang_names = get_language_names()
target_lang_names = [n for n in lang_names if n != "Auto Detect"]

# ----- Input card ---------------------------------------------------------
with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">INPUT LANGUAGE</div>', unsafe_allow_html=True)
    st.session_state.source_lang_name = st.selectbox(
        "Source language",
        options=lang_names,
        index=lang_names.index(st.session_state.source_lang_name)
        if st.session_state.source_lang_name in lang_names
        else 0,
        label_visibility="collapsed",
    )

    st.session_state.source_text = st.text_area(
        "Enter text",
        value=st.session_state.source_text,
        height=200,
        placeholder="Type or paste text to translate...",
        label_visibility="collapsed",
        max_chars=4000,
    )

    char_count = count_characters(st.session_state.source_text)
    st.markdown(f'<div class="char-counter">{char_count} / 4000 characters</div>', unsafe_allow_html=True)

    btn_cols = st.columns([1, 1])
    with btn_cols[0]:
        translate_clicked = st.button("✨ Translate", use_container_width=True, type="primary")
    with btn_cols[1]:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        clear_clicked = st.button("🧹 Clear", use_container_width=True, on_click=do_clear)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----- Swap button ----------------------------------------------------------
with col_swap:
    st.markdown("<div style='height: 130px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    st.button("⇄", help="Swap languages", on_click=do_swap_languages, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Output card -----------------------------------------------------------
with col_output:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">OUTPUT LANGUAGE</div>', unsafe_allow_html=True)
    default_target_index = (
        target_lang_names.index(st.session_state.target_lang_name)
        if st.session_state.target_lang_name in target_lang_names
        else 0
    )
    st.session_state.target_lang_name = st.selectbox(
        "Target language",
        options=target_lang_names,
        index=default_target_index,
        label_visibility="collapsed",
    )

    output_placeholder = st.empty()
    if st.session_state.translated_text:
        output_placeholder.markdown(
            f'<div class="output-box">{st.session_state.translated_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        output_placeholder.markdown(
            '<div class="output-box" style="color:#64748B;">Your translation will appear here...</div>',
            unsafe_allow_html=True,
        )

    out_cols = st.columns(3)
    with out_cols[0]:
        st.download_button(
            "📄 TXT",
            data=build_txt_bytes(
                st.session_state.source_text,
                st.session_state.translated_text,
                st.session_state.source_lang_name,
                st.session_state.target_lang_name,
            ),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not st.session_state.translated_text,
        )
    with out_cols[1]:
        pdf_bytes = None
        if st.session_state.translated_text:
            pdf_bytes = build_pdf_bytes(
                st.session_state.source_text,
                st.session_state.translated_text,
                st.session_state.source_lang_name,
                st.session_state.target_lang_name,
            )
        st.download_button(
            "🧾 PDF",
            data=pdf_bytes or b"",
            file_name="translation.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not pdf_bytes,
        )
    with out_cols[2]:
        if st.button("📋 Copy", use_container_width=True, disabled=not st.session_state.translated_text):
            st.toast("Translation copied! (select text above and Ctrl+C)", icon="📋")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# Perform translation
# ==========================================================================
if translate_clicked:
    text_to_translate = st.session_state.source_text.strip()
    if not text_to_translate:
        st.toast("Please enter some text to translate.", icon="⚠️")
    else:
        src_code = code_from_name(st.session_state.source_lang_name)
        tgt_code = code_from_name(st.session_state.target_lang_name)

        with st.spinner("🧠 Running Transformer inference..."):
            try:
                result = translate(text_to_translate, src_code, tgt_code)
                st.session_state.translated_text = result.translated_text
                st.session_state.last_model_used = result.model_used
                st.session_state.last_processing_time = result.processing_time
                st.session_state.last_detected = result.detected_language

                detected_name = (
                    name_from_code(result.detected_language) if result.detected_language else None
                )
                effective_src_name = detected_name or st.session_state.source_lang_name

                add_history_entry(
                    source_lang=effective_src_name,
                    target_lang=st.session_state.target_lang_name,
                    source_text=text_to_translate,
                    translated_text=result.translated_text,
                    model_used=result.model_used,
                )
                st.toast("Translation complete!", icon="✅")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"⚠️ Translation failed: {exc}\n\n"
                    "This can happen on first run while the model downloads, "
                    "or if there's no internet connection to fetch the model from "
                    "Hugging Face Hub. Please try again."
                )

# ==========================================================================
# Translation statistics
# ==========================================================================
st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Translation Statistics</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Characters", count_characters(st.session_state.source_text))
with s2:
    st.metric("Words", count_words(st.session_state.source_text))
with s3:
    st.metric("Reading Time", estimate_reading_time(st.session_state.source_text))
with s4:
    st.metric(
        "Processing Time",
        format_processing_time(st.session_state.last_processing_time)
        if st.session_state.last_processing_time
        else "—",
    )

if st.session_state.last_detected:
    st.markdown(
        f'<span class="pill">Detected language: {name_from_code(st.session_state.last_detected)}</span>',
        unsafe_allow_html=True,
    )

# ==========================================================================
# About the AI model
# ==========================================================================
st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)

with st.expander("🧬 About the AI Model", expanded=False):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"""
            **Model in use:** `{st.session_state.last_model_used}`

            **Architecture:** Transformer (Encoder–Decoder, Attention-based)

            **Frameworks:** Hugging Face Transformers + PyTorch
            """
        )
    with a2:
        st.markdown(
            """
            **Primary engine:** MarianMT (`Helsinki-NLP/opus-mt-*`)
            — fast, specialized bilingual models.

            **Fallback engine:** Facebook M2M100 (`facebook/m2m100_418M`)
            — a single multilingual model covering 100 languages,
            used automatically when no dedicated MarianMT pair exists.
            """
        )
    st.markdown(
        """
        <div style="margin-top:10px;">
            <span class="pill">Hugging Face 🤗</span>
            <span class="pill">PyTorch 🔥</span>
            <span class="pill">Local Inference</span>
            <span class="pill">No External API Calls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# Footer
# ==========================================================================
st.markdown(
    '<div class="app-footer">Built with ❤️ using Streamlit, Hugging Face Transformers & PyTorch</div>',
    unsafe_allow_html=True,
)
