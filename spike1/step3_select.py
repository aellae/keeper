"""
STEP 3 — Il selettore

    score(i)   = w_q·qualità + w_f·volti + w_u·unicità + w_b·comportamento
    scelta     = greedy MMR:  argmax [ score(i) − λ · max_sim(i, già_selezionate) ]

Il punto (§4.2 spec): il RANKING produce 60 tramonti identici. La SELEZIONE con
vincolo di diversità produce un album. λ è la manopola varietà ↔ qualità pura.

    python step3_select.py day_A --quota 15 --lam 0.5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, normalize, load_favorites

# Pesi dello score. §4.3: i volti e l'unicità stanno SOPRA l'estetica.
# NOTA 2026-07-12: su PhotoKoyasan (giornata di tempio, quasi zero volti)
# reweighting verso exposure (0.65/0.05/0.15/0.15, quality 0.2/0.6/0.2) è stato
# testato e NON ha migliorato il recall (2/12 in entrambi i casi, solo hit diversi) —
# con 12 esempi il tuning dei pesi rischia di inseguire rumore. Pesi originali ripristinati.
W_QUALITY = 0.25   # spareggio, non criterio
W_FACES   = 0.35   # le persone battono i tramonti
W_UNIQUE  = 0.25   # il soggetto raro > il dodicesimo torii
W_BEHAV   = 0.15   # favorites: il giudizio già espresso dall'utente


def build_scores(names, embs, meta, favorites):
    sharp = normalize([np.log1p(m["sharpness"]) for m in meta])
    expo = np.array([m["exposure"] for m in meta])
    contr = normalize([m["contrast"] for m in meta])
    quality = 0.5 * sharp + 0.3 * np.clip(expo, 0, 1) + 0.2 * contr

    # volti: presenza + area, saturata (3 volti non valgono il triplo di 1)
    n_faces = np.array([min(m["n_faces"], 3) / 3.0 for m in meta])
    f_area = normalize([m["face_area"] for m in meta])
    faces = 0.6 * n_faces + 0.4 * f_area

    # unicità: quanto sei lontana dal resto della giornata
    S = embs @ embs.T
    np.fill_diagonal(S, -1.0)
    unique = normalize(1.0 - S.max(axis=1))

    behav = np.array([1.0 if n in favorites else 0.0 for n in names])

    score = (W_QUALITY * quality + W_FACES * faces
             + W_UNIQUE * unique + W_BEHAV * behav)
    return normalize(score), dict(quality=quality, faces=faces, unique=unique, behav=behav)


def mmr(scores, embs, quota, lam):
    """Greedy: massimizza score penalizzando la somiglianza con le già scelte."""
    S = embs @ embs.T
    n = len(scores)
    selected = []
    for _ in range(min(quota, n)):
        best, best_val = None, -1e9
        for i in range(n):
            if i in selected:
                continue
            penalty = max((S[i, j] for j in selected), default=0.0)
            val = scores[i] - lam * penalty
            if val > best_val:
                best, best_val = i, val
        selected.append(best)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quota", type=int, default=15)
    ap.add_argument("--lam", type=float, default=0.5)
    a = ap.parse_args()

    d = np.load(out_path(a.folder, "features.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    embs = d["embs"]
    meta = json.loads(out_path(a.folder, "meta.json").read_text())
    favorites = load_favorites(a.folder)

    scores, parts = build_scores(names, embs, meta, favorites)
    picked = mmr(scores, embs, a.quota, a.lam)

    out = {
        "quota": a.quota, "lambda": a.lam,
        "selected": [names[i] for i in picked],
        "detail": [
            {
                "name": names[i], "score": round(float(scores[i]), 3),
                "quality": round(float(parts["quality"][i]), 3),
                "faces": round(float(parts["faces"][i]), 3),
                "unique": round(float(parts["unique"][i]), 3),
            }
            for i in picked
        ],
    }
    out_path(a.folder, "selection.json").write_text(json.dumps(out, indent=1))

    print(f"[ok] selection.json  (quota={a.quota}, λ={a.lam})")
    for r in out["detail"]:
        print(f"     {r['name']:<24} score={r['score']:.2f}  "
              f"q={r['quality']:.2f} volti={r['faces']:.2f} uniq={r['unique']:.2f}")
    if favorites:
        print(f"\n     ({len(favorites)} preferiti letti da favorites.txt)")


if __name__ == "__main__":
    main()
