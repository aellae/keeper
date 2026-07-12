"""
STEP 7 — IL NUMERO dello spike-2

Applica il modello addestrato sul giorno A alle coppie del giorno B (mai visto).
Domanda: data una coppia dello stesso cluster di B, il modello indovina quale hai tenuto?

CANCELLO G2 — doppia condizione, decisa PRIMA di vedere i numeri:
    A) accuracy coppie su B  ≥ 75%
    B) le coppie devono BATTERE il controllo binario (task 2.5)

    ✅ entrambe        → il modello a due livelli funziona (§5.3.5)
    ⚠️  A sì, B no     → funziona ma le coppie sono inutili: semplifica, usa le binarie
    ❌ A no (~50%)     → hai imparato Kyoto, non l'utente. Rivedi le feature.

    python step7_eval_taste.py day_A day_B
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path
# (le feature del test si costruiscono in load_day con la scala del training)


class LoadedPCA:
    """Ricostruisce la trasformazione PCA salvata (senza ri-fittare)."""
    def __init__(self, components, mean):
        self.components_ = components
        self.mean_ = mean

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


def pair_accuracy_bt(w, X, name_idx, pairs):
    """Bradley-Terry: a batte b se w·(f(a)−f(b)) > 0."""
    correct = total = 0
    for win, lose in pairs:
        if win not in name_idx or lose not in name_idx:
            continue
        d = X[name_idx[win]] - X[name_idx[lose]]
        margin = float(w @ d)
        if margin != 0.0:
            correct += margin > 0
            total += 1
        else:  # pareggio esatto: mezzo punto (non barare né a favore né contro)
            correct += 0.5
            total += 1
    return (correct / total if total else 0.0), total


def pair_accuracy_binary(w, b, X, name_idx, pairs):
    """Controllo: a batte b se P_tengo(a) > P_tengo(b). Con logit monotona basta w·f."""
    correct = total = 0
    for win, lose in pairs:
        if win not in name_idx or lose not in name_idx:
            continue
        ua = float(w @ X[name_idx[win]])
        ub = float(w @ X[name_idx[lose]])
        if ua != ub:
            correct += ua > ub
            total += 1
        else:
            correct += 0.5
            total += 1
    return (correct / total if total else 0.0), total


def load_day(folder, pca, scaler):
    d = np.load(out_path(folder, "features.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    embs = d["embs"]
    meta = json.loads(out_path(folder, "meta.json").read_text())
    # ⚠️ stessa standardizzazione e stessa PCA del TRAINING: mai ri-fittare sul test
    mu, sd = scaler
    raw = np.array([[np.log1p(m["sharpness"]), m["exposure"], m["contrast"],
                     min(m["n_faces"], 3) / 3.0, m["face_area"]] for m in meta])
    X = np.hstack([(raw - mu) / sd, pca.transform(embs)])
    pairs = json.loads(out_path(folder, "pairs.json").read_text())["pairs"]
    return names, X, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_folder", help="giorno di training (day_A)")
    ap.add_argument("test_folder", help="giorno di test, MAI VISTO (day_B)")
    a = ap.parse_args()

    if a.train_folder == a.test_folder:
        print("[!!] Stai testando sul giorno di training: il numero sarà gonfiato")
        print("     e NON risponde alla domanda dello spike-2 (generalizzazione).\n")

    m = np.load(out_path(a.train_folder, "taste_model.npz"), allow_pickle=True)
    pca = LoadedPCA(m["pca_components"], m["pca_mean"])
    scaler = (m["expl_mu"], m["expl_sd"])

    names_B, X_B, pairs_B = load_day(a.test_folder, pca, scaler)
    idx_B = {n: i for i, n in enumerate(names_B)}

    acc_bt, n_bt = pair_accuracy_bt(m["w_bt"], X_B, idx_B, pairs_B)
    acc_bin, _ = pair_accuracy_binary(m["w_bin"], m["b_bin"], X_B, idx_B, pairs_B)

    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  TRAIN: {a.train_folder}   →   TEST: {a.test_folder}")
    print(f"  coppie di test valutate: {n_bt}")
    print(bar)
    print(f"  caso (50%)                    accuracy   50.0%")
    print(f"  controllo BINARIO (task 2.5)  accuracy  {acc_bin:6.1%}")
    print(f"  ▶ BRADLEY-TERRY (coppie)      accuracy  {acc_bt:6.1%}")
    print(bar)

    cond_A = acc_bt >= 0.75
    cond_B = acc_bt > acc_bin

    if cond_A and cond_B:
        v = "✅  G2 APERTO — il gusto GENERALIZZA e le coppie battono le binarie.\n      Il modello a due livelli (§5.3.5) si costruisce."
    elif cond_A and not cond_B:
        v = "⚠️   G2 PARZIALE — generalizza, ma le coppie NON battono le binarie.\n      La tesi §5.3.3 non regge su questi dati: SEMPLIFICA, usa le binarie."
    elif acc_bt <= 0.60:
        v = "❌  G2 CHIUSO — ~caso: hai imparato il giorno A, non l'utente.\n      Rivedi le feature prima di costruirci sopra."
    else:
        v = "⚠️   G2 DEBOLE — generalizza poco (60-75%).\n      Servono più contesto o più feature. Non costruire ancora."
    print(f"  {v}")
    print(bar)

    if n_bt < 20:
        print("  [!] Meno di 20 coppie di test: il numero è rumoroso.")
        print("      Considera un giorno B più denso, o accorpa due giorni di test.\n")

    out_path(a.test_folder, "taste_result.json").write_text(json.dumps({
        "train": a.train_folder, "test": a.test_folder,
        "n_pairs_test": n_bt,
        "acc_bradley_terry": acc_bt, "acc_binary_control": acc_bin,
        "gate_A_generalizes": bool(cond_A), "gate_B_pairs_beat_binary": bool(cond_B),
    }, indent=1))
    print(f"[ok] {a.test_folder}/taste_result.json\n")


if __name__ == "__main__":
    main()
