"""
STEP 4 — IL NUMERO

    recall@quota  contro la tua ground truth.
    + baseline (random / solo qualità / uno per gruppo)
    + sweep di λ
    + DIAGNOSI dei due modi di sbagliare

    python step4_evaluate.py day_A --quota 15

CANCELLO G1, deciso PRIMA di vedere il numero:
    ≥ 12/15  →  il selettore vive
    8–11/15  →  serve il modello di gusto (spike-2)
    ≤  7/15  →  approccio sbagliato. Fermati.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth, load_favorites
from step3_select import build_scores, mmr


def recall(picked_names, truth):
    hit = len(set(picked_names) & truth)
    return hit / max(len(truth), 1), hit


def baseline_random(names, quota, seed=0):
    rng = random.Random(seed)
    return rng.sample(names, min(quota, len(names)))


def baseline_one_per_cluster(names, bursts, scores, name_idx, quota):
    """Il migliore di ogni gruppo, poi i gruppi ordinati per score del loro campione."""
    reps = []
    for members in bursts.values():
        idxs = [name_idx[m] for m in members]
        best = max(idxs, key=lambda i: scores[i])
        reps.append(best)
    reps.sort(key=lambda i: scores[i], reverse=True)
    return [names[i] for i in reps[:quota]]


def diagnose(picked_names, truth, bursts, name_to_burst):
    """
    Due modi di sbagliare, e distinguono se serve spike-2:
      A) GRUPPO MANCATO      → priorità sui gruppi sbagliate. Le feature sono da rifare.
      B) GRUPPO GIUSTO,      → i gruppi sono giusti, manca il GUSTO.
         FOTO SBAGLIATA         Spike-2 è esattamente la cura.
    """
    picked = set(picked_names)
    picked_bursts = {name_to_burst[n] for n in picked_names if n in name_to_burst}

    missed_group, wrong_photo = [], []
    for t in truth:
        if t in picked:
            continue
        b = name_to_burst.get(t)
        if b is not None and b in picked_bursts:
            chosen = [n for n in picked_names if name_to_burst.get(n) == b]
            wrong_photo.append((t, chosen))
        else:
            missed_group.append(t)
    return missed_group, wrong_photo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quota", type=int, default=15)
    a = ap.parse_args()

    truth = load_ground_truth(a.folder)
    quota = a.quota or len(truth)

    d = np.load(out_path(a.folder, "features.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]

    # sanity: la ground truth deve riferirsi a foto realmente processate
    unknown = truth - set(names)
    if unknown:
        print(f"[!!] {len(unknown)} nomi della ground truth NON esistono tra le foto processate:")
        for u in sorted(unknown):
            print(f"     - {u}")
        print("     (typo? file rinominato? cartella diversa?) — li escludo dal conteggio.\n")
        truth = truth - unknown
        if not truth:
            raise SystemExit("[stop] ground truth vuota dopo l'esclusione.")

    embs = d["embs"]
    meta = json.loads(out_path(a.folder, "meta.json").read_text())
    clusters = json.loads(out_path(a.folder, "clusters.json").read_text())
    bursts = clusters["bursts"]
    favorites = load_favorites(a.folder)

    name_idx = {n: i for i, n in enumerate(names)}
    name_to_burst = {n: b for b, members in bursts.items() for n in members}

    scores, _ = build_scores(names, embs, meta, favorites)

    # ---------- sweep di λ ----------
    lams = np.linspace(0.0, 1.0, 21)
    curve = []
    for lam in lams:
        sel = [names[i] for i in mmr(scores, embs, quota, float(lam))]
        r, h = recall(sel, truth)
        curve.append((float(lam), r, h))
    best_lam, best_r, best_h = max(curve, key=lambda x: x[1])

    # ---------- baseline ----------
    rnd = np.mean([recall(baseline_random(names, quota, s), truth)[0] for s in range(20)])
    only_q = curve[0]                       # λ = 0  → puro ranking di qualità
    opc = baseline_one_per_cluster(names, bursts, scores, name_idx, quota)
    opc_r, opc_h = recall(opc, truth)

    # ---------- report ----------
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  GROUND TRUTH: {len(truth)} foto scelte a mano · quota algoritmo: {quota}")
    print(bar)
    print(f"  random                        recall  {rnd:5.1%}")
    print(f"  solo qualità (λ=0)            recall  {only_q[1]:5.1%}   ({only_q[2]}/{len(truth)})")
    print(f"  uno per gruppo                recall  {opc_r:5.1%}   ({opc_h}/{len(truth)})")
    print(f"  ▶ MMR  (λ={best_lam:.2f})              recall  {best_r:5.1%}   ({best_h}/{len(truth)})")
    print(bar)

    # il confronto che conta davvero
    if best_r > only_q[1] + 1e-9:
        print(f"  ✅ la diversità AIUTA: +{(best_r - only_q[1]):.1%} sul ranking puro")
    else:
        print(f"  ❌ la diversità NON aiuta. La tesi §4.2 della spec è sbagliata su questi dati.")
    print(bar)

    # ---------- CANCELLO (soglie proporzionali: 12/15 = 80%, 8/15 ≈ 53%) ----------
    ratio = best_h / max(len(truth), 1)
    if ratio >= 0.80:
        verdict = f"✅  G1 APERTO — il selettore vive ({best_h}/{len(truth)}). Vai allo spike-2."
    elif ratio >= 0.53:
        verdict = f"⚠️   G1 PARZIALE ({best_h}/{len(truth)}) — serve il modello di gusto. Spike-2 è la cura."
    else:
        verdict = f"❌  G1 CHIUSO ({best_h}/{len(truth)}) — approccio sbagliato. Non passare oltre. Rivedi le feature."
    print(f"  {verdict}")
    print(bar)

    # ---------- diagnosi ----------
    sel_best = [names[i] for i in mmr(scores, embs, quota, best_lam)]
    missed, wrong = diagnose(sel_best, truth, bursts, name_to_burst)
    print(f"\n  DIAGNOSI DEGLI ERRORI  ({len(truth) - best_h} foto tue non prese)")
    print(f"    gruppo mancato ............ {len(missed):>2}   → priorità sui gruppi sbagliate")
    print(f"    gruppo giusto, foto sbagliata {len(wrong):>2}   → manca il GUSTO (spike-2)")
    if wrong:
        print("\n    Esempi 'foto sbagliata' (tu → algoritmo, stesso gruppo):")
        for t, chosen in wrong[:5]:
            print(f"      {t}  →  {', '.join(chosen)}")
    if missed:
        print(f"\n    Gruppi mancati del tutto: {', '.join(missed[:8])}")

    if len(wrong) > len(missed):
        print("\n    ➜ La maggioranza degli errori è 'foto sbagliata nel gruppo giusto'.")
        print("      Il selettore funziona. Quello che manca è il tuo gusto. SPIKE-2.")
    elif missed:
        print("\n    ➜ Stai mancando gruppi interi: il problema è a monte del gusto.")
        print("      Rivedi i pesi in step3 (volti? unicità?) prima di spike-2.")

    # ---------- grafico ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o", lw=1.5)
        plt.axhline(0.80, ls="--", c="green", label="soglia G1 (80%)")
        plt.axhline(only_q[1], ls=":", c="gray", label="solo qualità (λ=0)")
        plt.axvline(best_lam, ls="--", c="red", alpha=.4)
        plt.xlabel("λ  (0 = solo qualità · 1 = massima varietà)")
        plt.ylabel(f"recall@{quota}")
        plt.title("Quanto conta il vincolo di diversità")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        p = out_path(a.folder, "lambda_sweep.png")
        plt.savefig(p, dpi=130)
        print(f"\n[ok] {p}")
    except ImportError:
        pass

    out_path(a.folder, "result.json").write_text(json.dumps({
        "truth": sorted(truth), "quota": quota,
        "best_lambda": best_lam, "recall": best_r, "hits": best_h,
        "baseline_random": rnd, "baseline_quality_only": only_q[1],
        "baseline_one_per_cluster": opc_r,
        "missed_group": missed, "wrong_photo_right_group": [w[0] for w in wrong],
        "selected": sel_best,
    }, indent=1))
    print(f"[ok] result.json\n")


if __name__ == "__main__":
    main()
