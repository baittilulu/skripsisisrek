"""
WisataBajo - Sistem Rekomendasi Wisata Labuan Bajo (TF-IDF + Sentence-BERT)
Logika pra-pemrosesan, TF-IDF, SBERT, dan hybrid weighted concatenation
disalin dari notebook penelitian tanpa perubahan perhitungan.
"""
import base64
import html
import io
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CSV_NAME = "wisata_labuan_bajo_dataset.csv"
IMG_DIR = BASE_DIR / "images"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

ALPHA = 0.6          # bobot TF-IDF   (sama dengan notebook)
BETA = 0.4           # bobot S-BERT   (sama dengan notebook)
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 10           # maksimal destinasi yang ditampilkan
MIN_SCORE = 0.25     # ambang cosine similarity minimum agar dianggap "ditemukan"

POPULAR = [
    ("Snorkeling", "fa-water"),
    ("Pantai", "fa-umbrella-beach"),
    ("Hiking", "fa-person-hiking"),
    ("Komodo", "fa-dragon"),
    ("Budaya", "fa-landmark"),
]

st.set_page_config(page_title="WisataBajo", page_icon="🗺️", layout="centered",
                   initial_sidebar_state="collapsed")


# ----------------------------------------------------------------------
# ENGINE (identik dengan notebook)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat dataset & model (hanya sekali)…")
def load_engine():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from sentence_transformers import SentenceTransformer

    csv_path = next((p for p in (BASE_DIR / "data" / CSV_NAME, BASE_DIR / CSV_NAME) if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"File {CSV_NAME} tidak ditemukan di folder data/.")
    df = pd.read_csv(csv_path)

    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()

    text_columns = ["Nama Wisata", "Kategori", "Deskripsi", "Tipe", "Aktivitas", "Lokasi"]
    # str() per sel: sel kosong jadi "nan" (sama dengan notebook, aman di pandas 2 maupun 3)
    df["teks_gabungan"] = df[text_columns].apply(lambda c: c.map(str)).agg(" ".join, axis=1)

    def preprocess_leksikal(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = stopword.remove(text)
        text = stemmer.stem(text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_semantik(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    df["teks_clean_leksikal"] = df["teks_gabungan"].apply(preprocess_leksikal)
    df["teks_clean_semantik"] = df["teks_gabungan"].apply(preprocess_semantik)

    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["teks_clean_leksikal"])
    tfidf_vectors = tfidf_matrix.toarray()

    sbert_model = SentenceTransformer(SBERT_MODEL)
    sbert_vectors = sbert_model.encode(df["teks_clean_semantik"].tolist(), show_progress_bar=False)
    sbert_vectors_norm = sbert_vectors / np.linalg.norm(sbert_vectors, axis=1, keepdims=True)

    final_vectors_hybrid = np.hstack((ALPHA * tfidf_vectors, BETA * sbert_vectors_norm))

    nama_bersih = df["Nama Wisata"].map(str).apply(lambda x: x.split(",")[0].strip())

    return dict(df=df, stemmer=stemmer, stopword=stopword, tfidf=tfidf,
                sbert_model=sbert_model, hybrid=final_vectors_hybrid,
                nama=nama_bersih, preprocess_semantik=preprocess_semantik)


def get_recommendations(query_text, k=TOP_K, min_score=MIN_SCORE):
    """Metode 'hybrid' dari get_recommendations_engine di notebook."""
    E = load_engine()
    query_semantik = E["preprocess_semantik"](query_text)

    q_clean_base = str(query_text).lower().strip()
    q_clean_base = re.sub(r'[^a-z\s]', ' ', q_clean_base)
    q_no_stop = E["stopword"].remove(q_clean_base)
    query_leksikal = " ".join(E["stemmer"].stem(w) for w in q_no_stop.split())

    q_tfidf = E["tfidf"].transform([query_leksikal]).toarray()
    q_sbert = E["sbert_model"].encode([query_semantik], show_progress_bar=False)
    q_sbert_norm = q_sbert / np.linalg.norm(q_sbert)
    q_vector = np.hstack((ALPHA * q_tfidf, BETA * q_sbert_norm))

    scores = cosine_similarity(q_vector, E["hybrid"]).flatten()

    res = pd.DataFrame({
        "idx": np.arange(len(scores)),
        "Nama Wisata": E["nama"].values,
        "Nilai Similarity": scores,
    }).sort_values(by="Nilai Similarity", ascending=False).reset_index(drop=True)

    res = res[res["Nilai Similarity"] >= min_score]
    return res.head(k)


# ----------------------------------------------------------------------
# HELPER DATA / GAMBAR
# ----------------------------------------------------------------------
def clean(v, default="-"):
    s = str(v).strip()
    return default if s.lower() in ("", "nan", "none") else s


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@st.cache_data(show_spinner=False)
def img_data_uri(path, max_w):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_w, max_w * 4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def find_image(row, nama, max_w=900):
    cands = []
    if "Gambar" in row.index and clean(row["Gambar"], "") != "":
        cands.append(IMG_DIR / clean(row["Gambar"]))
    cands += [IMG_DIR / (slugify(nama) + ext) for ext in IMG_EXTS]
    for p in cands:
        if p.exists():
            return img_data_uri(str(p), max_w)
    return None


def find_background():
    for stem in ("background", "bg", "hero"):
        for ext in IMG_EXTS:
            p = IMG_DIR / (stem + ext)
            if p.exists():
                return img_data_uri(str(p), 1600)
    return None


def split_aktivitas(text):
    text = clean(text, "")
    if not text:
        return []
    parts = re.split(r",", text) if "," in text else text.split()
    return [p.strip() for p in parts if p.strip()]


def esc(x):
    return html.escape(str(x), quote=True)


# ----------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------
BASE_CSS = """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
html, body, [class*="css"], .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
a { text-decoration: none; }
@media print {
  .st-key-btn_other, .st-key-btn_print, iframe { display: none !important; }
}
</style>
"""

HOME_CSS = """
<style>
.stApp { %(bg)s background-size: cover; background-position: center; background-attachment: fixed; }
.block-container { max-width: 640px; padding-top: 3.5rem; }
.hero { text-align:center; color:#fff; margin-bottom: 1.6rem; }
.hero .logo { font-size: 2.3rem; font-weight: 800; display:flex; align-items:center; justify-content:center; gap:.6rem; }
.hero .logo i { background:#3498db; border-radius:12px; padding:.45rem .55rem; font-size:1.5rem; }
.hero .sub { color:#dfe6ee; font-size:.95rem; margin-top:.4rem; }
.st-key-searchcard { background:#fff; border-radius:18px; padding:2rem 2.2rem 1.6rem; box-shadow:0 12px 40px rgba(0,0,0,.35); }
.st-key-searchcard h2.ttl { text-align:center; color:#2c3e50; font-size:1.9rem; font-weight:700; margin:0 0 1.3rem; padding:0; }
.st-key-searchcard h2.ttl i { margin-right:.5rem; }
.st-key-searchcard .lbl { font-weight:700; color:#2c3e50; font-size:.95rem; margin-bottom:.4rem; }
.st-key-searchcard .lbl i { margin-right:.4rem; color:#555; }
.st-key-searchcard [data-testid="stForm"] { border:none; padding:0; }
.st-key-searchcard input { border-radius:10px !important; padding:.8rem 1rem !important; color:#333 !important; }
.st-key-searchcard [data-baseweb="input"], .st-key-searchcard [data-baseweb="base-input"] {
  border-radius:10px; border:1.5px solid #dfe3e8; background:#fff; }
.pop-lbl { text-align:center; color:#7f8c8d; font-size:.9rem; margin:1.1rem 0 .6rem; }
.pop-lbl i { color:#f1c40f; margin-right:.3rem; }
.chips { display:flex; flex-wrap:wrap; justify-content:center; gap:.55rem; margin-bottom:1.2rem; }
.chips a { background:#edf1f6; color:#34495e; border-radius:999px; padding:.45rem 1rem; font-size:.85rem; display:inline-flex; gap:.4rem; align-items:center; }
.chips a:hover { background:#dbe6f2; }
.st-key-searchcard [data-testid="stFormSubmitButton"] button {
  width:100%; border:none; border-radius:10px; padding:.85rem; color:#fff; font-weight:600; font-size:1rem;
  background:linear-gradient(90deg,#3d9be0,#2f80c9); }
.st-key-searchcard [data-testid="stFormSubmitButton"] button::before {
  content:"\\f002"; font-family:"Font Awesome 6 Free"; font-weight:900; margin-right:.55rem; }
.st-key-searchcard [data-testid="stFormSubmitButton"] button:hover { filter:brightness(1.07); color:#fff; }
.foot { text-align:center; color:#95a5a6; font-size:.8rem; margin-top:1.1rem; }
.foot i { margin-right:.3rem; }
</style>
"""

RESULT_CSS = """
<style>
.stApp { background:#f2f4f8; }
.block-container { max-width: 1000px; padding: 0 0 2.5rem 0 !important; }
.rband { background:linear-gradient(90deg,#3d9be0,#2f80c9); color:#fff; padding:1.6rem 2rem 1.3rem; display:flex; justify-content:space-between; align-items:flex-start; }
.rband h1 { color:#fff; font-size:1.7rem; font-weight:700; margin:0; padding:0; }
.rband .cnt { margin-top:.5rem; font-size:.95rem; opacity:.95; }
.rband .cnt i, .rband h1 i { margin-right:.45rem; }
.btn-home { background:#fff; color:#2c3e50 !important; border-radius:8px; padding:.45rem .9rem; font-size:.85rem; font-weight:600; }
.btn-home i { margin-right:.35rem; }
.grid { display:grid; grid-template-columns:repeat(2,1fr); gap:1.5rem; padding:1.5rem 2rem; }
@media (max-width:700px){ .grid { grid-template-columns:1fr; padding:1rem; } }
.card { background:#fff; border-radius:14px; box-shadow:0 4px 18px rgba(0,0,0,.08); overflow:hidden; display:flex; flex-direction:column; }
.card .pic { width:100%; height:175px; object-fit:cover; display:block; border-bottom:3px solid #3d9be0; background:linear-gradient(135deg,#a8c0d8,#6d8fb0); }
.card .ph { width:100%; height:175px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:2.5rem; border-bottom:3px solid #3d9be0; background:linear-gradient(135deg,#a8c0d8,#6d8fb0); }
.card .body { padding:1.1rem 1.2rem 1rem; display:flex; flex-direction:column; gap:.65rem; flex:1; }
.card h3 { margin:0; padding:0; font-size:1.25rem; font-weight:700; color:#2c3e50; }
.tag { align-self:flex-start; background:#ececec; color:#555; border-radius:8px; padding:.2rem .6rem; font-size:.75rem; }
.tag i { margin-right:.3rem; }
.card .desc { color:#444; font-size:.9rem; line-height:1.45; }
.card .act { color:#7f8c8d; font-size:.85rem; font-style:italic; }
.card .act i { margin-right:.4rem; font-style:normal; }
.card .loc { color:#2f80c9; font-size:.85rem; font-weight:600; }
.card .loc i { margin-right:.4rem; }
.card .foot { display:flex; justify-content:flex-end; margin-top:auto; }
.btn-detail { background:#3d9be0; color:#fff !important; border-radius:8px; padding:.5rem .95rem; font-size:.85rem; font-weight:600; }
.btn-detail i { margin-right:.35rem; }
.btn-detail:hover { background:#2f80c9; }
.empty { text-align:center; padding:3rem 1rem; color:#7f8c8d; }
</style>
"""

DETAIL_CSS = """
<style>
.stApp { background:#f2f4f8; }
.block-container { max-width: 1000px; padding: 0 0 2.5rem 0 !important; }
.dhead { background:linear-gradient(90deg,#667eea,#764ba2); color:#fff; padding:1.6rem 2rem; display:flex; justify-content:space-between; align-items:flex-start; }
.dhead h1 { color:#fff; margin:0 0 .6rem; padding:0; font-size:2rem; font-weight:700; }
.badge { display:inline-block; background:rgba(0,0,0,.28); color:#fff; border-radius:6px; padding:.18rem .55rem; font-size:.7rem; margin-right:.4rem; }
.badge i { margin-right:.3rem; }
.btn-home { background:#fff; color:#2c3e50 !important; border-radius:8px; padding:.45rem .9rem; font-size:.85rem; font-weight:600; }
.btn-home i { margin-right:.35rem; }
.dgrid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; padding:1.5rem 2rem 1rem; align-items:start; }
@media (max-width:760px){ .dgrid { grid-template-columns:1fr; padding:1rem; } }
.dimg { width:100%; border-radius:14px; display:block; object-fit:cover; min-height:300px; max-height:420px; box-shadow:0 4px 18px rgba(0,0,0,.12); }
.dph { width:100%; height:330px; border-radius:14px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:3rem; background:linear-gradient(135deg,#a8c0d8,#6d8fb0); }
.right { display:flex; flex-direction:column; gap:1.1rem; }
.dcard { background:#fff; border-radius:14px; padding:1.2rem 1.4rem; box-shadow:0 4px 18px rgba(0,0,0,.08); }
.dcard h2 { margin:0 0 .9rem; padding:0 0 .55rem; font-size:1.2rem; font-weight:700; color:#2c3e50; border-bottom:2px solid #667eea; }
.dcard h2 i { color:#667eea; margin-right:.5rem; }
.dcard p { margin:0; color:#444; font-size:.9rem; line-height:1.55; }
.dcard table { width:100%; border-collapse:collapse; font-size:.88rem; }
.dcard td { padding:.55rem .2rem; border-bottom:1px solid #eef0f3; color:#444; }
.dcard td:first-child { font-weight:700; color:#2c3e50; width:35%; }
.dcard tr:last-child td { border-bottom:none; }
.chip { display:inline-block; background:#e9eef0; color:#444; border-radius:999px; padding:.3rem .8rem; font-size:.8rem; margin:0 .4rem .4rem 0; }
.chip i { color:#27ae60; margin-right:.3rem; font-size:.7rem; }
.st-key-btn_other button, .st-key-btn_print button { border:none; border-radius:8px; color:#fff; font-weight:600; padding:.55rem 1rem; width:100%; }
.st-key-btn_other button { background:#1d8cf8; }
.st-key-btn_print button { background:#1e9e5a; }
.st-key-btn_other button:hover, .st-key-btn_print button:hover { filter:brightness(1.08); color:#fff; }
.st-key-btn_other button::before { content:"\\f002"; font-family:"Font Awesome 6 Free"; font-weight:900; margin-right:.5rem; }
.st-key-btn_print button::before { content:"\\f02f"; font-family:"Font Awesome 6 Free"; font-weight:900; margin-right:.5rem; }
</style>
"""


# ----------------------------------------------------------------------
# HALAMAN
# ----------------------------------------------------------------------
def go(**params):
    st.query_params.clear()
    for k, v in params.items():
        st.query_params[k] = str(v)
    st.rerun()


def page_home():
    bg = find_background()
    bg_css = (f"background-image: linear-gradient(rgba(10,25,45,.55), rgba(10,25,45,.75)), url('{bg}');"
              if bg else "background: linear-gradient(160deg,#0f2a44 0%,#1d4f73 55%,#2a6f8f 100%);")
    st.markdown(HOME_CSS.replace("%(bg)s", bg_css), unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><div class="logo"><i class="fa-solid fa-map-location-dot"></i>WisataBajo</div>'
        '<div class="sub">Sistem Rekomendasi Wisata Labuan Bajo</div></div>',
        unsafe_allow_html=True)

    with st.container(key="searchcard"):
        st.markdown('<h2 class="ttl"><i class="fa-solid fa-magnifying-glass"></i>Cari Destinasi Wisata</h2>'
                    '<div class="lbl"><i class="fa-regular fa-keyboard"></i>Masukkan kata kunci:</div>',
                    unsafe_allow_html=True)
        with st.form("search_form", border=False):
            q = st.text_input("kata kunci", label_visibility="collapsed",
                              placeholder="Contoh: snorkeling, pantai, komodo, budaya, hiking")
            chips = "".join(
                f'<a href="?page=results&amp;q={quote(label)}" target="_self"><i class="fa-solid {icon}"></i>{label}</a>'
                for label, icon in POPULAR)
            st.markdown(f'<div class="pop-lbl"><i class="fa-solid fa-bolt"></i>Pencarian Populer:</div>'
                        f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Cari Rekomendasi")
        st.markdown('<div class="foot"><i class="fa-solid fa-circle-info"></i>'
                    'Sistem menggunakan algoritma TF-IDF + SBERT</div>', unsafe_allow_html=True)

    if submitted:
        if q.strip():
            go(page="results", q=q.strip())
        else:
            st.warning("Masukkan kata kunci terlebih dahulu.")


def page_results(query):
    st.markdown(RESULT_CSS, unsafe_allow_html=True)
    E = load_engine()
    res = get_recommendations(query)

    st.markdown(
        f'<div class="rband"><div><h1><i class="fa-solid fa-magnifying-glass"></i>Hasil Pencarian: "{esc(query)}"</h1>'
        f'<div class="cnt"><i class="fa-solid fa-chart-line"></i>Ditemukan {len(res)} destinasi</div></div>'
        f'<a class="btn-home" href="?" target="_self"><i class="fa-solid fa-house"></i>Beranda</a></div>',
        unsafe_allow_html=True)

    if res.empty:
        st.markdown('<div class="empty"><i class="fa-regular fa-face-frown" style="font-size:2rem"></i>'
                    '<p>Tidak ada destinasi yang cocok. Coba kata kunci lain.</p></div>', unsafe_allow_html=True)
        return

    cards = []
    for _, r in res.iterrows():
        i = int(r["idx"])
        row = E["df"].iloc[i]
        nama = E["nama"].iloc[i]
        uri = find_image(row, nama, 700)
        pic = (f'<img class="pic" src="{uri}" alt="{esc(nama)}">' if uri
               else '<div class="ph"><i class="fa-regular fa-image"></i></div>')
        tag = f'{esc(clean(row["Kategori"]))} | {esc(clean(row["Tipe"]))}'
        cards.append(
            f'<div class="card">{pic}<div class="body"><h3>{esc(nama)}</h3>'
            f'<span class="tag"><i class="fa-solid fa-tag"></i>{tag}</span>'
            f'<div class="desc">{esc(clean(row["Deskripsi"]))}</div>'
            f'<div class="act"><i class="fa-solid fa-person-running"></i>{esc(clean(row["Aktivitas"]))}</div>'
            f'<div class="loc"><i class="fa-solid fa-location-dot"></i>{esc(clean(row["Lokasi"]))}</div>'
            f'<div class="foot"><a class="btn-detail" href="?page=detail&amp;id={i}&amp;q={quote(query)}" target="_self">'
            f'<i class="fa-solid fa-circle-info"></i>Detail</a></div></div></div>')
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def page_detail(idx, query):
    st.markdown(DETAIL_CSS, unsafe_allow_html=True)
    E = load_engine()
    if idx < 0 or idx >= len(E["df"]):
        go()
    row = E["df"].iloc[idx]
    nama = E["nama"].iloc[idx]
    uri = find_image(row, nama, 1100)
    pic = (f'<img class="dimg" src="{uri}" alt="{esc(nama)}">' if uri
           else '<div class="dph"><i class="fa-regular fa-image"></i></div>')

    chips = "".join(f'<span class="chip"><i class="fa-solid fa-check"></i>{esc(a)}</span>'
                    for a in split_aktivitas(row["Aktivitas"])) or '<p>-</p>'

    st.markdown(
        f'<div class="dhead"><div><h1>{esc(nama)}</h1>'
        f'<span class="badge"><i class="fa-solid fa-tag"></i>{esc(clean(row["Kategori"]))}</span>'
        f'<span class="badge"><i class="fa-solid fa-location-dot"></i>{esc(clean(row["Lokasi"]))}</span></div>'
        f'<a class="btn-home" href="?" target="_self"><i class="fa-solid fa-house"></i>Beranda</a></div>'
        f'<div class="dgrid"><div>{pic}</div><div class="right">'
        f'<div class="dcard"><h2><i class="fa-solid fa-circle-info"></i>Deskripsi</h2><p>{esc(clean(row["Deskripsi"]))}</p></div>'
        f'<div class="dcard"><h2><i class="fa-solid fa-list"></i>Informasi</h2><table>'
        f'<tr><td>Kategori</td><td>{esc(clean(row["Kategori"]))}</td></tr>'
        f'<tr><td>Tipe</td><td>{esc(clean(row["Tipe"]))}</td></tr>'
        f'<tr><td>Lokasi</td><td>{esc(clean(row["Lokasi"]))}</td></tr></table></div>'
        f'<div class="dcard"><h2><i class="fa-solid fa-person-running"></i>Aktivitas</h2>{chips}</div>'
        f'</div></div>', unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([2.6, 2.4, 1.3, 2.6])
    with c1:
        if st.button("Cari Destinasi Lain", key="btn_other"):
            go()
    with c2:
        if st.button("Cetak", key="btn_print"):
            components.html("<script>window.parent.print();</script>", height=0)


# ----------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------
st.markdown(BASE_CSS, unsafe_allow_html=True)
qp = st.query_params
page = qp.get("page", "home")
query = qp.get("q", "")

try:
    if page == "results" and query.strip():
        page_results(query.strip())
    elif page == "detail" and str(qp.get("id", "")).isdigit():
        page_detail(int(qp["id"]), query)
    else:
        page_home()
except FileNotFoundError as e:
    st.error(str(e))
