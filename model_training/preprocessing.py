"""
preprocessing.py  —  Pipeline Preprocessing Dataset Resep Indonesia
====================================================================
Membaca raw CSV dari Kaggle, membersihkan data, membangun TF-IDF matrix,
dan menyimpan semua artifact yang dibutuhkan oleh recommender.py.

Output (disimpan ke folder `data/`):
  - recipes_clean.csv        : dataset bersih siap pakai
  - tfidf_vectorizer.pkl     : TF-IDF vectorizer (sudah di-fit)
  - tfidf_matrix.npz         : sparse matrix TF-IDF semua resep
  - ingredient_vocab.json    : vocab bahan untuk autocomplete

Cara jalankan:
  python preprocessing.py --input Indonesian_Food_Recipes.csv
  python preprocessing.py --input Indonesian_Food_Recipes.csv --data-dir data
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Kamus satuan & noise words (sama persis dgn recommender.py)
# ─────────────────────────────────────────────
SATUAN = {
    "ml","liter","l","cc","sdm","sdt","sendok","makan","teh","gelas","cangkir","cup",
    "gram","gr","g","kg","kilogram","ons",
    "buah","biji","butir","siung","lembar","lbr","batang","ikat","genggam",
    "iris","potong","ptg","ruas","cm","helai","lonjor","tangkai","bonggol",
    "ekor","pcs","bungkus","sachet","kotak","kaleng","porsi","mangkok",
    "pak","papan","tusuk",
}
NOISE_WORDS = {
    "secukupnya","sesuai","selera","sedikit","banyak","kurang","lebih",
    "halus","kasar","iris","cincang","geprek","memarkan","potong",
    "rebus","goreng","tumis","sangrai","bakar","kukus","haluskan",
    "opsional","optional","boleh","skip","atau","dan","dengan",
    "untuk","jika","kalau","bisa","pakai","merk","merek",
    "kira","saja","aja","ya","yaa","deh","lah","sesukanya",
}


# ─────────────────────────────────────────────
# Helper: normalisasi satu string bahan
# ─────────────────────────────────────────────

def normalize_ingredient(raw: str) -> str:
    """
    Bersihkan satu nama bahan dari noise (satuan, angka, kata2 deskriptif).
    Logika ini HARUS identik dengan RecipeRecommender._normalize_ingredient().
    """
    text = raw.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)           # hapus teks dalam kurung
    text = re.sub(r"[¼½¾⅓⅔⅛⅜⅝⅞]", "", text)    # hapus simbol pecahan
    text = re.sub(r"\b\d+[\d/.,]*\b", "", text)  # hapus angka
    text = re.sub(r"[^a-z\s]", " ", text)        # hapus karakter non-huruf
    tokens = text.split()
    tokens = [t for t in tokens if t not in SATUAN and t not in NOISE_WORDS and len(t) > 1]
    return " ".join(tokens).strip()


# ─────────────────────────────────────────────
# Helper: parse kolom ingredients menjadi list
# ─────────────────────────────────────────────

def parse_ingredients_raw(ingredients_str: str) -> list[str]:
    """
    Ingredients di CSV dipisah tanda '--'.
    Kembalikan list string per bahan (setelah strip).
    """
    if not isinstance(ingredients_str, str):
        return []
    parts = [p.strip() for p in ingredients_str.split("--")]
    return [p for p in parts if p]


def parse_ingredients_clean(cleaned_str: str) -> list[str]:
    """
    Ingredients Cleaned di CSV dipisah koma.
    Kembalikan list setelah normalisasi ulang.
    """
    if not isinstance(cleaned_str, str):
        return []
    parts = [p.strip() for p in cleaned_str.split(",")]
    normalized = [normalize_ingredient(p) for p in parts]
    return [n for n in normalized if n]


# ─────────────────────────────────────────────
# Helper: parse steps menjadi list kalimat
# ─────────────────────────────────────────────

def parse_steps(steps_str: str) -> list[str]:
    """
    Steps dipisah \\n, lalu filter baris yang tidak kosong.
    """
    if not isinstance(steps_str, str):
        return []
    lines = steps_str.split("\\n")
    cleaned = [l.strip() for l in lines if l.strip()]
    return cleaned


# ─────────────────────────────────────────────
# Helper: estimasi waktu memasak dari teks steps
# ─────────────────────────────────────────────

def estimate_time_minutes(steps_str: str, n_steps: int) -> int:
    """
    Estimasi waktu memasak (dalam menit) dari teks steps.
    Strategi:
      1. Cari angka menit/jam yang disebutkan di teks.
      2. Fallback: n_steps * 7 menit per langkah (estimasi sederhana).
    """
    if not isinstance(steps_str, str):
        return n_steps * 7

    text = steps_str.lower()

    # Cari pola seperti "45 menit", "1 jam", "30 mnt"
    menit_matches = re.findall(r"(\d+(?:[,\.]\d+)?)\s*(?:menit|mnt|min)", text)
    jam_matches   = re.findall(r"(\d+(?:[,\.]\d+)?)\s*(?:jam|hours?|hrs?)", text)

    total = 0
    for m in menit_matches:
        try:
            total += float(m.replace(",", "."))
        except ValueError:
            pass
    for j in jam_matches:
        try:
            total += float(j.replace(",", ".")) * 60
        except ValueError:
            pass

    # Kalau tidak ditemukan waktu eksplisit, fallback ke heuristik
    if total == 0:
        total = n_steps * 7

    # Cap minimal 5 menit, maksimal 360 menit
    return int(max(5, min(360, total)))


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_preprocessing(input_path: str, data_dir: str = "data") -> None:
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CSV ────────────────────────────────────────────────
    log.info(f"Membaca dataset: {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"  Total baris awal: {len(df):,}")

    # ── 2. Rename kolom ────────────────────────────────────────────
    df = df.rename(columns={
        "Title":               "title",
        "Ingredients":         "ingredients_str",
        "Steps":               "steps_str",
        "Loves":               "loves",
        "URL":                 "URL",
        "Category":            "kategori",
        "Title Cleaned":       "title_clean",
        "Total Ingredients":   "total_ingredients",
        "Ingredients Cleaned": "ingredients_cleaned_str",
        "Total Steps":         "steps_count",
    })

    # ── 3. Drop duplikat & null kritis ─────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["title"])
    df = df.dropna(subset=["title", "ingredients_str", "steps_str"])
    log.info(f"  Setelah drop duplikat/null: {len(df):,} (dihapus {before - len(df):,})")

    # ── 4. Isi title_clean yang kosong ─────────────────────────────
    df["title_clean"] = df["title_clean"].fillna(df["title"].str.lower())

    # ── 5. Parse ingredients ───────────────────────────────────────
    log.info("Parsing & normalisasi ingredients...")
    df["ingredients_raw"] = df["ingredients_str"].apply(parse_ingredients_raw)
    df["ingredients_clean"] = df["ingredients_cleaned_str"].apply(parse_ingredients_clean)

    # Fallback: kalau ingredients_clean kosong, parse dari ingredients_str
    mask_empty = df["ingredients_clean"].apply(len) == 0
    df.loc[mask_empty, "ingredients_clean"] = df.loc[mask_empty, "ingredients_str"].apply(
        lambda s: [normalize_ingredient(p) for p in parse_ingredients_raw(s) if normalize_ingredient(p)]
    )
    log.info(f"  Rows dengan ingredients_clean kosong: {df['ingredients_clean'].apply(len).eq(0).sum()}")

    # ── 6. Buat string TF-IDF per resep ────────────────────────────
    # Gabungkan semua bahan bersih menjadi satu string per resep
    df["ingredients_tfidf"] = df["ingredients_clean"].apply(lambda lst: " ".join(lst))

    # ── 7. Parse steps ─────────────────────────────────────────────
    log.info("Parsing steps...")
    df["steps_list"] = df["steps_str"].apply(parse_steps)

    # ── 8. Estimasi waktu memasak ──────────────────────────────────
    log.info("Estimasi waktu memasak...")
    df["estimated_time_min"] = df.apply(
        lambda row: estimate_time_minutes(row["steps_str"], row["steps_count"]),
        axis=1,
    )
    log.info(f"  Rata-rata estimasi waktu: {df['estimated_time_min'].mean():.1f} menit")

    # ── 9. Reset index jadi recipe_id ─────────────────────────────
    df = df.reset_index(drop=True)
    df.insert(0, "recipe_id", df.index)

    # ── 10. Pilih kolom final ──────────────────────────────────────
    df_clean = df[[
        "recipe_id", "title", "title_clean", "kategori",
        "ingredients_raw", "ingredients_clean", "ingredients_tfidf",
        "steps_list", "steps_count",
        "estimated_time_min", "loves", "URL",
    ]].copy()

    # ── 11. Simpan recipes_clean.csv ──────────────────────────────
    csv_path = output_dir / "recipes_clean.csv"
    df_clean.to_csv(csv_path, index=False)
    log.info(f"Disimpan: {csv_path} ({len(df_clean):,} resep)")

    # ── 12. Build TF-IDF matrix ───────────────────────────────────
    log.info("Membangun TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        ngram_range=(1, 2),     # unigram + bigram (misal "bawang merah" jadi satu token)
        min_df=2,               # abaikan bahan yang muncul < 2 resep
        max_df=0.95,            # abaikan bahan yang muncul di > 95% resep (terlalu umum)
        sublinear_tf=True,      # log(tf) + 1 supaya frekuensi tinggi tidak dominan
    )

    tfidf_matrix = vectorizer.fit_transform(df_clean["ingredients_tfidf"])
    log.info(f"  Shape matrix: {tfidf_matrix.shape}  (resep × vocab)")
    log.info(f"  Vocab size: {len(vectorizer.vocabulary_):,} token")

    # Simpan vectorizer
    vec_path = output_dir / "tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    log.info(f"Disimpan: {vec_path}")

    # Simpan sparse matrix
    mat_path = output_dir / "tfidf_matrix.npz"
    sparse.save_npz(str(mat_path), tfidf_matrix)
    log.info(f"Disimpan: {mat_path}")

    # ── 13. Build ingredient vocab (untuk autocomplete) ───────────
    log.info("Membangun ingredient vocab...")
    from collections import Counter
    ingredient_counter: Counter = Counter()
    for ing_list in df_clean["ingredients_clean"]:
        ingredient_counter.update(ing_list)

    # Urutkan dari yang paling sering muncul
    sorted_ingredients = [
        {"name": name, "count": count}
        for name, count in ingredient_counter.most_common()
    ]

    vocab_data = {
        "total_ingredients": len(sorted_ingredients),
        "ingredients": sorted_ingredients,
    }

    vocab_path = output_dir / "ingredient_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    log.info(f"Disimpan: {vocab_path} ({len(sorted_ingredients):,} bahan unik)")

    # ── 14. Ringkasan akhir ────────────────────────────────────────
    log.info("")
    log.info("=" * 50)
    log.info("  PREPROCESSING SELESAI")
    log.info("=" * 50)
    log.info(f"  Total resep       : {len(df_clean):,}")
    log.info(f"  Kategori          : {df_clean['kategori'].nunique()}")
    for kat, cnt in df_clean["kategori"].value_counts().items():
        log.info(f"    - {kat:<10}: {cnt:,}")
    log.info(f"  Vocab TF-IDF      : {len(vectorizer.vocabulary_):,} token")
    log.info(f"  Bahan unik        : {len(sorted_ingredients):,}")
    log.info(f"  Rata-rata bahan   : {df_clean['ingredients_clean'].apply(len).mean():.1f} per resep")
    log.info(f"  Estimasi waktu    : {df_clean['estimated_time_min'].mean():.0f} menit (rata-rata)")
    log.info("=" * 50)
    log.info(f"  Output folder     : {output_dir.resolve()}")
    log.info("=" * 50)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing dataset resep Indonesia untuk sistem rekomendasi."
    )
    parser.add_argument(
        "--input",
        default="Indonesian_Food_Recipes.csv",
        help="Path ke file CSV input (default: Indonesian_Food_Recipes.csv)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder output untuk artifact (default: data/)",
    )
    args = parser.parse_args()
    run_preprocessing(input_path=args.input, data_dir=args.data_dir)
