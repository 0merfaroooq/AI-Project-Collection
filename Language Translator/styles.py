"""
styles.py
---------
Custom CSS for the AI Language Translator app.

Implements a premium, dark, glassmorphism SaaS theme:
    Background : #0F172A
    Cards      : #1E293B
    Accent     : #38BDF8
    Text       : White

Also hides Streamlit's default menu / footer / header chrome so the
app doesn't feel like a stock Streamlit project.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ---------------------------------------------------------------- */
/* Global resets                                                     */
/* ---------------------------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden; height: 0;}
div[data-testid="stDecoration"] {visibility: hidden;}
div[data-testid="stStatusWidget"] {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at 15% 0%, #16213c 0%, #0F172A 45%, #0b1120 100%);
    color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background: #0B1220;
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}

/* ---------------------------------------------------------------- */
/* Typography                                                         */
/* ---------------------------------------------------------------- */
h1, h2, h3, .brand-font {
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
}

/* ---------------------------------------------------------------- */
/* Hero section                                                       */
/* ---------------------------------------------------------------- */
.hero-wrap {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem 1rem;
    animation: fadeIn 0.9s ease-in-out;
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38BDF8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 60%, #7DD3FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    line-height: 1.15;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto;
    font-weight: 400;
}

/* ---------------------------------------------------------------- */
/* Glass cards                                                        */
/* ---------------------------------------------------------------- */
.glass-card {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeIn 0.7s ease-in-out;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.35);
}

.metric-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.85));
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 18px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 26px rgba(56, 189, 248, 0.18);
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #38BDF8;
    font-family: 'Sora', sans-serif;
}

.metric-label {
    color: #94A3B8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-label {
    color: #7DD3FC;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

/* ---------------------------------------------------------------- */
/* Buttons                                                             */
/* ---------------------------------------------------------------- */
div.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #38BDF8 0%, #0EA5E9 100%);
    color: #0B1220;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35);
}

div.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 22px rgba(56, 189, 248, 0.5);
    color: #0B1220;
}

div.stButton > button:active {
    transform: translateY(0px) scale(0.99);
}

/* Secondary / ghost buttons (swap, clear) */
.ghost-btn button {
    background: transparent !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    box-shadow: none !important;
}

/* ---------------------------------------------------------------- */
/* Inputs                                                              */
/* ---------------------------------------------------------------- */
.stTextArea textarea, .stTextInput input {
    background: rgba(15, 23, 42, 0.7) !important;
    color: #F8FAFC !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    font-size: 0.98rem !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}

.char-counter {
    text-align: right;
    color: #64748B;
    font-size: 0.78rem;
    margin-top: -6px;
}

/* ---------------------------------------------------------------- */
/* Output box                                                          */
/* ---------------------------------------------------------------- */
.output-box {
    min-height: 190px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px dashed rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 1rem;
    color: #E2E8F0;
    font-size: 1rem;
    line-height: 1.55;
    white-space: pre-wrap;
}

/* ---------------------------------------------------------------- */
/* Badges / pills                                                      */
/* ---------------------------------------------------------------- */
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    color: #7DD3FC;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}

/* ---------------------------------------------------------------- */
/* History items                                                       */
/* ---------------------------------------------------------------- */
.history-item {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    transition: border-color 0.2s ease;
}
.history-item:hover {
    border-color: rgba(56, 189, 248, 0.4);
}
.history-lang {
    color: #38BDF8;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
}

/* ---------------------------------------------------------------- */
/* Animations                                                          */
/* ---------------------------------------------------------------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }
    100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Skeleton loading shimmer */
.skeleton {
    background: linear-gradient(90deg, rgba(148,163,184,0.08) 25%, rgba(148,163,184,0.18) 37%, rgba(148,163,184,0.08) 63%);
    background-size: 400% 100%;
    animation: shimmer 1.4s ease infinite;
    border-radius: 10px;
    height: 18px;
    margin-bottom: 8px;
}
@keyframes shimmer {
    0% { background-position: 100% 50%; }
    100% { background-position: 0 50%; }
}

/* Divider */
.soft-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.25), transparent);
    margin: 1.6rem 0;
    border: none;
}

/* Footer credit */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem 0;
}
</style>
"""


def inject_custom_css(st) -> None:
    """Inject the custom CSS block into the given Streamlit module instance."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
