"""
STEP 3b — Selettore con taste-centroid (esperimento dopo G1/G2 chiusi)

G1 e G2 hanno mostrato, su due giornate indipendenti e in entrambe le direzioni
train/test, che lo score basato su qualità tecnica (nitidezza/esposizione/contrasto)
e su "unicità = lontananza dalla media della giornata" non correla col gusto reale.
I pesi imparati (spike-2) cambiavano perfino segno da un giorno all'altro.

Due cose invece hanno funzionato: gli embedding CLIP (informativi, il problema era
COME venivano usati) e il meccanismo di ridondanza di MMR — penalizzare la somiglianza
alle foto GIÀ scelte nella selezione finale, cosa diversa dal "premiare la rarità
globale" (che è quanto falliva).

Nuova ipotesi: invece di "quanto sei tecnicamente buona" o "quanto sei rara oggi",
score = quanto assomigli, nell'embedding CLIP, alle foto che questa utente ha
TENUTO IN UN'ALTRA GIORNATA (taste centroid = media normalizzata degli embedding
delle foto scelte). Zero parametri da overfittare — solo una media — adatto a un
numero di esempi troppo piccolo per una logistic regression (vedi G2).

    python step3b_taste_centroid.py --train ../PhotoKoyasan --test ../PhotoOsaka --quota 15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth, normalize
from step3_select import mmr
from step4_evaluate import recall, baseline_random, baseline_one_per_cluster, diagnose


def load_day(folder):
    d = np.load(out_path(folder, "features.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    embs = d["embs"]  # già normalizzati (norma 1) in step1
    clusters = json.loads(out_path(folder, "clusters.json").read_text())
    return names, embs, clusters["bursts"]


def taste_centroid(train_names, train_embs, train_truth):
    idx = [i for i, n in enumerate(train_names) if n in train_truth]
    if not idx:
        raise SystemExit("[stop] nessuna foto della ground truth trovata tra le feature del train.")
    c = train_embs[idx].mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="giorno da cui calcolare il taste centroid")
    ap.add_argument("--test", required=True, help="giorno su cui valutare (mai visto dal centroid)")
    ap.add_argument("--quota", type=int, default=15)
    ap.add_argument("--train-truth", help="JSON alternativo per il centroid (es. il set 'declutter' più ampio, invece del ground_truth.json stretto)")
    ap.add_argument("--center", action="store_true",
                     help="sottrae la media della giornata da ogni embedding prima del confronto — "
                          "isola 'quanto preferisco QUESTA rispetto al resto della giornata', non "
                          "'quanto assomiglia genericamente a un tempio/città'")
    a = ap.parse_args()

    train_names, train_embs, _ = load_day(a.train)
    if a.train_truth:
        train_truth = set(json.loads(Path(a.train_truth).read_text())["selected"])
    else:
        train_truth = load_ground_truth(a.train)

    test_names, test_embs, test_bursts = load_day(a.test)
    test_truth = load_ground_truth(a.test)

    if a.center:
        train_embs = train_embs - train_embs.mean(axis=0, keepdims=True)
        test_embs = test_embs - test_embs.mean(axis=0, keepdims=True)
        train_embs = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-9)
        test_embs = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-9)

    centroid = taste_centroid(train_names, train_embs, train_truth)
    name_idx = {n: i for i, n in enumerate(test_names)}
    name_to_burst = {n: b for b, members in test_bursts.items() for n in members}

    raw_sim = test_embs @ centroid  # coseno diretto: embedding già normalizzati
    scores = normalize(raw_sim)

    lams = np.linspace(0.0, 1.0, 21)
    curve = []
    for lam in lams:
        sel = [test_names[i] for i in mmr(scores, test_embs, a.quota, float(lam))]
        r, h = recall(sel, test_truth)
        curve.append((float(lam), r, h))
    best_lam, best_r, best_h = max(curve, key=lambda x: x[1])

    rnd = np.mean([recall(baseline_random(test_names, a.quota, s), test_truth)[0] for s in range(20)])
    pure_rank = curve[0]  # lam=0 -> puro ranking per similarità al centroid, senza MMR
    opc = baseline_one_per_cluster(test_names, test_bursts, scores, name_idx, a.quota)
    opc_r, opc_h = recall(opc, test_truth)

    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  TASTE CENTROID   train={a.train}   test={a.test}")
    print(f"  ground truth test: {len(test_truth)} foto · quota: {a.quota}")
    print(bar)
    print(f"  random                          recall  {rnd:5.1%}")
    print(f"  solo similarità al centroid      recall  {pure_rank[1]:5.1%}   ({pure_rank[2]}/{len(test_truth)})")
    print(f"  uno per gruppo                   recall  {opc_r:5.1%}   ({opc_h}/{len(test_truth)})")
    print(f"  ▶ MMR + centroid (λ={best_lam:.2f})        recall  {best_r:5.1%}   ({best_h}/{len(test_truth)})")
    print(bar)

    sel_best = [test_names[i] for i in mmr(scores, test_embs, a.quota, best_lam)]
    missed, wrong = diagnose(sel_best, test_truth, test_bursts, name_to_burst)
    print(f"  gruppo mancato ............ {len(missed):>2}")
    print(f"  gruppo giusto, foto sbagliata {len(wrong):>2}")
    print(bar)

    out_path(a.test, "result_taste_centroid.json").write_text(json.dumps({
        "train": a.train, "test": a.test, "quota": a.quota,
        "best_lambda": best_lam, "recall": best_r, "hits": best_h,
        "baseline_random": rnd, "baseline_pure_centroid": pure_rank[1],
        "baseline_one_per_cluster": opc_r,
        "selected": sel_best, "truth": sorted(test_truth),
    }, indent=1))
    print(f"[ok] {a.test}/result_taste_centroid.json\n")


if __name__ == "__main__":
    main()
