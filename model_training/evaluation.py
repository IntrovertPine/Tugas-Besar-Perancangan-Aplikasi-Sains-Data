"""
evaluation.py  —  Evaluasi Sistem Rekomendasi Resep (Leave-One-Out)
====================================================================
Mengukur performa sistem rekomendasi menggunakan pendekatan Leave-One-Out:

  1. Ambil N sampel resep dari dataset secara acak
  2. Untuk setiap resep, sembunyikan 1 bahan (bahan pertama / bahan acak)
  3. Gunakan sisa bahan sebagai input ke sistem rekomendasi
  4. Cek apakah resep asli muncul di top-K hasil rekomendasi
  5. Hitung Precision@K, Recall@K, F1-Score, Accuracy

Cara jalankan (dari folder root project):
  python model_training/evaluation.py
  python model_training/evaluation.py --sample 500 --top-k 10
  python model_training/evaluation.py --sample 500 --top-k 10 --seed 99
"""

from __future__ import annotations

import argparse
import ast
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Tambahkan root project ke sys.path agar bisa import recommender
_this = Path(__file__).resolve()
ROOT = _this.parent.parent if _this.parent.name == "model_training" else _this.parent
sys.path.insert(0, str(ROOT))

from recommender import RecipeRecommender

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
# Fungsi evaluasi utama
# ─────────────────────────────────────────────

def evaluate_leave_one_out(
    recommender: RecipeRecommender,
    sample_size: int = 300,
    top_k: int = 10,
    seed: int = 42,
    min_ingredients: int = 3,
) -> dict:
    """
    Evaluasi sistem dengan pendekatan Leave-One-Out.

    Args:
        recommender   : Instance RecipeRecommender yang sudah diload
        sample_size   : Jumlah resep yang dijadikan sampel evaluasi
        top_k         : Sistem diminta mengembalikan top-K rekomendasi
        seed          : Random seed untuk reprodusibilitas
        min_ingredients: Minimal jumlah bahan per resep agar layak dievaluasi

    Returns:
        dict berisi semua metrik evaluasi
    """
    random.seed(seed)
    np.random.seed(seed)

    df = recommender.df.copy()

    # ── 1. Filter resep yang punya cukup bahan ─────────────────────
    df["n_ing"] = df["ingredients_clean"].apply(len)
    df_eligible = df[df["n_ing"] >= min_ingredients].copy()
    log.info(f"Resep eligible (>= {min_ingredients} bahan): {len(df_eligible):,}")

    # ── 2. Ambil sampel acak ───────────────────────────────────────
    actual_sample = min(sample_size, len(df_eligible))
    sampled = df_eligible.sample(n=actual_sample, random_state=seed)
    log.info(f"Sampel evaluasi: {actual_sample:,} resep")

    # ── 3. Jalankan leave-one-out per resep ────────────────────────
    results = []
    hit_details = []   # untuk analisis per resep

    for idx, (_, row) in enumerate(sampled.iterrows()):
        ingredients = list(row["ingredients_clean"])
        recipe_id   = int(row["recipe_id"])
        title       = row["title"]

        # Pilih 1 bahan yang akan disembunyikan (acak)
        hidden_idx  = random.randint(0, len(ingredients) - 1)
        hidden_ing  = ingredients[hidden_idx]
        query_ings  = [ing for i, ing in enumerate(ingredients) if i != hidden_idx]

        # Jalankan rekomendasi
        recs = recommender.recommend(query_ings, top_n=top_k)
        recommended_ids = [r.recipe_id for r in recs]

        # Hit = resep asli muncul di top-K
        hit = recipe_id in recommended_ids
        rank = recommended_ids.index(recipe_id) + 1 if hit else None

        results.append(hit)
        hit_details.append({
            "recipe_id"  : recipe_id,
            "title"      : title,
            "n_ingredients": len(ingredients),
            "hidden_ing" : hidden_ing,
            "query_size" : len(query_ings),
            "hit"        : hit,
            "rank"       : rank,
        })

        # Progress log setiap 50 resep
        if (idx + 1) % 50 == 0:
            current_acc = sum(results) / len(results)
            log.info(f"  Progress: {idx+1}/{actual_sample} | Accuracy sementara: {current_acc:.1%}")

    # ── 4. Hitung metrik ───────────────────────────────────────────
    hits        = sum(results)
    total       = len(results)
    accuracy    = hits / total

    # Dalam konteks leave-one-out CBF:
    # - True Positive  (TP): resep asli muncul di top-K        → hit
    # - False Positive (FP): top-K terisi resep lain (bukan asli) → top_k - 1 kalau hit, top_k kalau miss
    # - False Negative (FN): resep asli tidak muncul           → miss
    # - True Negative  (TN): tidak relevan & tidak direkomendasikan → tidak bisa diukur (implicit feedback)
    #
    # Precision@K = TP / (TP + FP) ≈ proporsi rekomendasi yang "benar"
    # Karena hanya 1 ground truth per query:
    #   Precision@K = (1 jika hit else 0) / K
    # Recall@K    = TP / (TP + FN) = (1 jika hit else 0) / 1

    precision_at_k  = sum(1/top_k if h else 0 for h in results) / total
    recall_at_k     = accuracy   # karena ground truth per query = 1
    f1_score        = (
        2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) > 0 else 0.0
    )

    # Mean Reciprocal Rank (MRR) — bonus metrik
    reciprocal_ranks = [1/d["rank"] if d["rank"] else 0 for d in hit_details]
    mrr = sum(reciprocal_ranks) / total

    # Distribusi rank untuk resep yang hit
    ranks_hit = [d["rank"] for d in hit_details if d["rank"] is not None]

    metrics = {
        "sample_size"    : total,
        "top_k"          : top_k,
        "hits"           : hits,
        "accuracy"       : round(accuracy, 4),
        "precision_at_k" : round(precision_at_k, 4),
        "recall_at_k"    : round(recall_at_k, 4),
        "f1_score"       : round(f1_score, 4),
        "mrr"            : round(mrr, 4),
        "avg_rank_when_hit": round(np.mean(ranks_hit), 2) if ranks_hit else None,
    }

    return metrics, hit_details


# ─────────────────────────────────────────────
# Print laporan
# ─────────────────────────────────────────────

def print_report(metrics: dict, top_k: int) -> None:
    target = {
        "accuracy"       : 0.80,
        "precision_at_k" : 0.80,
        "recall_at_k"    : 0.75,
        "f1_score"       : 0.60,
    }

    print()
    print("=" * 58)
    print("  LAPORAN EVALUASI — Leave-One-Out")
    print("=" * 58)
    print(f"  Sampel resep    : {metrics['sample_size']:,}")
    print(f"  Top-K           : {top_k}")
    print(f"  Total hit       : {metrics['hits']:,} / {metrics['sample_size']:,}")
    print()
    print(f"  {'Metrik':<20} {'Skor':>8}   {'Target':>8}   {'Status'}")
    print(f"  {'-'*20}   {'-'*8}   {'-'*8}   {'-'*8}")

    for key, label in [
        ("accuracy",        "Accuracy"),
        ("precision_at_k",  f"Precision@{top_k}"),
        ("recall_at_k",     f"Recall@{top_k}"),
        ("f1_score",        "F1-Score"),
    ]:
        score  = metrics[key]
        tgt    = target.get(key, None)
        if tgt:
            status = "✓ LULUS" if score >= tgt else "✗ BELUM"
            print(f"  {label:<20} {score:>8.4f}   {tgt:>8.2f}   {status}")
        else:
            print(f"  {label:<20} {score:>8.4f}")

    print()
    print(f"  MRR (Mean Reciprocal Rank) : {metrics['mrr']:.4f}")
    if metrics["avg_rank_when_hit"]:
        print(f"  Rata-rata rank saat hit    : {metrics['avg_rank_when_hit']:.1f}")
    print("=" * 58)


# ─────────────────────────────────────────────
# Simpan detail hasil ke CSV
# ─────────────────────────────────────────────

def save_detail_csv(hit_details: list[dict], output_path: str) -> None:
    df_detail = pd.DataFrame(hit_details)
    df_detail.to_csv(output_path, index=False)
    log.info(f"Detail hasil evaluasi disimpan: {output_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi sistem rekomendasi resep dengan Leave-One-Out."
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Folder berisi artifact preprocessing (default: data/)"
    )
    parser.add_argument(
        "--sample", type=int, default=300,
        help="Jumlah resep sampel untuk evaluasi (default: 300)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K rekomendasi yang dievaluasi (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-detail", action="store_true",
        help="Simpan detail hasil per resep ke CSV"
    )
    args = parser.parse_args()

    # Resolve data dir relatif ke root project
    data_dir = ROOT / args.data_dir

    log.info("Memuat model rekomendasi...")
    rec = RecipeRecommender(data_dir=str(data_dir))

    log.info(f"Mulai evaluasi leave-one-out (sample={args.sample}, top_k={args.top_k})...")
    metrics, hit_details = evaluate_leave_one_out(
        recommender    = rec,
        sample_size    = args.sample,
        top_k          = args.top_k,
        seed           = args.seed,
        min_ingredients= 3,
    )

    print_report(metrics, top_k=args.top_k)

    if args.save_detail:
        out_path = ROOT / "model_training" / "evaluation_detail.csv"
        save_detail_csv(hit_details, str(out_path))


if __name__ == "__main__":
    main()
