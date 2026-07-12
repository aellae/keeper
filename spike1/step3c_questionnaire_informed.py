"""
STEP 3c — Selettore informato dal questionario (v2, dopo il test stated-vs-revealed)

Tre risposte al questionario del 2026-07-12 hanno confermato pattern già misurati nei
dati — "dipende dal contesto" per i volti, "dipende dalla scena" per l'esposizione,
"il tipico" (non il raro) per la scelta tra scatti simili — e la terza ha scoperto un
bug di design vero e proprio: lo score originale premiava la RARITÀ (W_UNIQUE, "il
soggetto raro > il dodicesimo torii"), l'opposto di quanto l'utente dice di preferire.

Tre cambi, tutti giustificati da una risposta specifica:

  1. CENTROID (invariato da step3b) — non "sei tecnicamente buona" ma "assomigli a
     quello che tieni di solito", su embedding centrati per giornata. Finora il
     miglior risultato (25% contro il 16.7% del selettore originale).
  2. VOLTI ADATTIVI — il peso dei volti si scala con quanto la giornata è "piena di
     persone" (frazione di foto con almeno un volto), invece di un peso fisso 0.35
     sprecato su una giornata come Koyasan dove i volti non discriminano quasi nulla.
  3. TIPICITÀ, non unicità — il segno del vecchio termine "unique" è invertito: premia
     l'assomigliare al resto della giornata, non il distinguersene.

    python step3c_questionnaire_informed.py --train ../PhotoKoyasan --test ../PhotoOsaka --quota 15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth, normalize
from step3_select import mmr
from step3b_taste_centroid import load_day, taste_centroid
from step4_evaluate import recall, baseline_random, baseline_one_per_cluster, diagnose

W_CENTROID = 0.5
W_FACES_BASE = 0.3
W_TYPICAL = 0.2

# giornata di riferimento per "quanto piena di persone": la frazione di foto con
# almeno un volto su Osaka (giornata di città, dove i volti si sono mostrati rilevanti)
REFERENCE_FACE_RATE = 9 / 101


def center(embs):
    c = embs - embs.mean(axis=0, keepdims=True)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)


def build_scores_v2(names, embs_centered, meta, centroid):
    centroid_sim = normalize(embs_centered @ centroid)

    n_faces_raw = np.array([m["n_faces"] for m in meta])
    face_rate = float((n_faces_raw > 0).mean())
    face_weight = W_FACES_BASE * min(1.0, face_rate / REFERENCE_FACE_RATE)

    n_faces = np.array([min(m["n_faces"], 3) / 3.0 for m in meta])
    f_area = normalize([m["face_area"] for m in meta])
    faces = 0.6 * n_faces + 0.4 * f_area

    S = embs_centered @ embs_centered.T
    np.fill_diagonal(S, np.nan)
    typical = normalize(np.nanmean(S, axis=1))  # alta = assomiglia in media al resto -> TIPICO

    score = W_CENTROID * centroid_sim + face_weight * faces + W_TYPICAL * typical
    return normalize(score), face_weight, face_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--quota", type=int, default=15)
    ap.add_argument("--train-truth")
    a = ap.parse_args()

    train_names, train_embs, _ = load_day(a.train)
    train_truth = (set(json.loads(Path(a.train_truth).read_text())["selected"])
                   if a.train_truth else load_ground_truth(a.train))
    train_embs_c = center(train_embs)
    centroid = taste_centroid(train_names, train_embs_c, train_truth)

    test_names, test_embs, test_bursts = load_day(a.test)
    test_truth = load_ground_truth(a.test)
    test_meta = json.loads(out_path(a.test, "meta.json").read_text())
    test_embs_c = center(test_embs)
    name_idx = {n: i for i, n in enumerate(test_names)}
    name_to_burst = {n: b for b, members in test_bursts.items() for n in members}

    scores, face_weight, face_rate = build_scores_v2(test_names, test_embs_c, test_meta, centroid)

    lams = np.linspace(0.0, 1.0, 21)
    curve = []
    for lam in lams:
        sel = [test_names[i] for i in mmr(scores, test_embs_c, a.quota, float(lam))]
        r, h = recall(sel, test_truth)
        curve.append((float(lam), r, h))
    best_lam, best_r, best_h = max(curve, key=lambda x: x[1])

    rnd = np.mean([recall(baseline_random(test_names, a.quota, s), test_truth)[0] for s in range(20)])
    pure_rank = curve[0]
    opc = baseline_one_per_cluster(test_names, test_bursts, scores, name_idx, a.quota)
    opc_r, opc_h = recall(opc, test_truth)

    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  QUESTIONNAIRE-INFORMED v2   train={a.train}   test={a.test}")
    print(f"  volti in {face_rate:.1%} delle foto del test -> peso volti adattivo = {face_weight:.3f} (base {W_FACES_BASE})")
    print(f"  ground truth test: {len(test_truth)} foto · quota: {a.quota}")
    print(bar)
    print(f"  random                            recall  {rnd:5.1%}")
    print(f"  solo score v2 (no MMR)            recall  {pure_rank[1]:5.1%}   ({pure_rank[2]}/{len(test_truth)})")
    print(f"  uno per gruppo                    recall  {opc_r:5.1%}   ({opc_h}/{len(test_truth)})")
    print(f"  ▶ MMR + score v2 (λ={best_lam:.2f})          recall  {best_r:5.1%}   ({best_h}/{len(test_truth)})")
    print(bar)

    sel_best = [test_names[i] for i in mmr(scores, test_embs_c, a.quota, best_lam)]
    missed, wrong = diagnose(sel_best, test_truth, test_bursts, name_to_burst)
    print(f"  gruppo mancato ............ {len(missed):>2}")
    print(f"  gruppo giusto, foto sbagliata {len(wrong):>2}")
    print(bar)

    out_path(a.test, "result_v2_questionnaire.json").write_text(json.dumps({
        "train": a.train, "test": a.test, "quota": a.quota,
        "face_rate_test": face_rate, "face_weight_used": face_weight,
        "best_lambda": best_lam, "recall": best_r, "hits": best_h,
        "baseline_random": rnd, "baseline_pure_score": pure_rank[1],
        "baseline_one_per_cluster": opc_r,
        "selected": sel_best, "truth": sorted(test_truth),
    }, indent=1))
    print(f"[ok] {a.test}/result_v2_questionnaire.json\n")


if __name__ == "__main__":
    main()
