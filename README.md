# WisataBajo – Sistem Rekomendasi Wisata Labuan Bajo

Streamlit app (TF-IDF + Sentence-BERT hybrid, alpha 0.6 / beta 0.4).

## Struktur
```
app.py
requirements.txt
.streamlit/config.toml
data/wisata_labuan_bajo_dataset.csv   <- taruh dataset (lihat data/README.txt)
images/                               <- taruh gambar (lihat images/README.txt)
```

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push ke GitHub, lalu di https://share.streamlit.io pilih repo, branch, dan file `app.py`.
