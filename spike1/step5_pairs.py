"""
STEP 5 — Estrazione delle coppie intra-cluster  (spike-2)

Dalla ground truth di un giorno + i suoi cluster, estrae le preferenze a coppie:
se nel cluster del torii (12 foto) hai scelto la n.3, escono 11 coppie
    (foto_3 > foto_k)   per ogni k ≠ 3
Il contenuto è costante dentro il cluster → le coppie isolano lo STILE (§5.3.3).

Prerequisiti (dalla pipeline spike-1, sulla stessa cartella):
    features.npz · meta.json · clusters.json · ground_truth.json

    python step5_pairs.py day_A        → day_A/pairs.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()

    truth = load_ground_truth(a.folder)
    clusters = json.loads(out_path(a.folder, "clusters.json").read_text())
    bursts = clusters["bursts"]

    pairs = []           # (vincitrice, perdente)
    used_clusters = 0
    skipped_multi = 0    # cluster con PIÙ scelte tue: ogni scelta batte solo le non-scelte

    for members in bursts.values():
        if len(members) < 2:
            continue
        winners = [m for m in members if m in truth]
        losers = [m for m in members if m not in truth]
        if not winners or not losers:
            continue
        used_clusters += 1
        if len(winners) > 1:
            skipped_multi += 1
        for w in winners:
            for l in losers:
                pairs.append([w, l])

    if not pairs:
        raise SystemExit(
            "[stop] Nessuna coppia estraibile: la ground truth non interseca nessun\n"
            "       cluster multi-foto. O i cluster sono sbagliati (rivedi step2),\n"
            "       o le tue scelte cadono tutte su foto singole."
        )

    out = {"n_pairs": len(pairs), "n_clusters_used": used_clusters, "pairs": pairs}
    out_path(a.folder, "pairs.json").write_text(json.dumps(out, indent=1))

    print(f"[ok] pairs.json — {len(pairs)} coppie da {used_clusters} cluster")
    if skipped_multi:
        print(f"     ({skipped_multi} cluster con più di una tua scelta: "
              f"ogni scelta batte le non-scelte, le scelte NON si battono tra loro)")
    if len(pairs) < 30:
        print("     [!] Poche coppie: il modello sarà rumoroso. Considera una giornata più densa.")


if __name__ == "__main__":
    main()
