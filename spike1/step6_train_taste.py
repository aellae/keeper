"""
STEP 6 — Bradley-Terry sulla differenza dei feature vector  (spike-2)

Il modello (§5.3.4 spec):
    utilità di una foto:   u(x) = w · f(x)
    P(a batte b)         = sigmoid( u(a) − u(b) ) = sigmoid( w · (f(a) − f(b)) )

cioè una LOGISTIC REGRESSION sulla DIFFERENZA dei feature vector, senza intercetta
(l'antisimmetria è strutturale: P(a>b) = 1 − P(b>a)).

Feature per foto = [qualità, volti, exif-derivate] + embedding CLIP ridotto via PCA.
La PCA serve perché con poche decine di coppie 512 dimensioni sono overfitting garantito.

Addestra ANCHE il controllo (task 2.5): stesso identico feature vector, ma
classificazione binaria tengo/butto per foto. Se a step7 le coppie non lo battono,
la tesi §5.3.3 è sbagliata.

    python step6_train_taste.py day_A          → day_A/taste_model.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth

PCA_DIM = 12          # dimensioni dell'embedding dopo la riduzione
C_REG = 0.5           # regolarizzazione L2 (più basso = più forte): poche coppie, serve


def photo_features(names, embs, meta, pca=None, fit_pca=False):
    """Feature vector per foto: [qualità/volti espliciti] ++ [embedding ridotto]."""
    explicit = np.array([
        [
            np.log1p(m["sharpness"]),
            m["exposure"],
            m["contrast"],
            min(m["n_faces"], 3) / 3.0,
            m["face_area"],
        ]
        for m in meta
    ], dtype=float)
    # standardizza le esplicite (le differenze devono essere su scale confrontabili)
    mu, sd = explicit.mean(0), explicit.std(0) + 1e-9

    if fit_pca:
        pca = PCA(n_components=min(PCA_DIM, embs.shape[0], embs.shape[1]))
        pca.fit(embs)
    reduced = pca.transform(embs) if pca is not None else np.zeros((len(names), 0))

    X = np.hstack([(explicit - mu) / sd, reduced])
    return X, pca, (mu, sd)


def train_bradley_terry(X, name_idx, pairs):
    """
    Ogni coppia (w, l) produce DUE esempi simmetrici:
        f(w) − f(l) → 1        f(l) − f(w) → 0
    fit_intercept=False: il modello DEVE essere antisimmetrico.
    """
    D, y = [], []
    for w, l in pairs:
        if w not in name_idx or l not in name_idx:
            continue
        d = X[name_idx[w]] - X[name_idx[l]]
        D.append(d);  y.append(1)
        D.append(-d); y.append(0)
    D, y = np.array(D), np.array(y)
    clf = LogisticRegression(fit_intercept=False, C=C_REG, max_iter=2000)
    clf.fit(D, y)
    train_acc = clf.score(D, y)
    return clf.coef_[0], train_acc, len(D) // 2


def train_binary_control(X, names, truth):
    """Controllo (task 2.5): classificatore tengo/butto per foto, stesse feature."""
    y = np.array([1 if n in truth else 0 for n in names])
    if y.sum() == 0 or y.sum() == len(y):
        return None, 0.0
    clf = LogisticRegression(C=C_REG, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf, clf.score(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="giorno di TRAINING (day_A)")
    a = ap.parse_args()

    d = np.load(out_path(a.folder, "features.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    embs = d["embs"]
    meta = json.loads(out_path(a.folder, "meta.json").read_text())
    pairs = json.loads(out_path(a.folder, "pairs.json").read_text())["pairs"]
    truth = load_ground_truth(a.folder)
    name_idx = {n: i for i, n in enumerate(names)}

    X, pca, (mu, sd) = photo_features(names, embs, meta, fit_pca=True)

    w_bt, acc_bt, n_pairs = train_bradley_terry(X, name_idx, pairs)
    clf_bin, acc_bin = train_binary_control(X, names, truth)

    np.savez(
        out_path(a.folder, "taste_model.npz"),
        w_bt=w_bt,
        w_bin=clf_bin.coef_[0] if clf_bin is not None else np.zeros_like(w_bt),
        b_bin=clf_bin.intercept_ if clf_bin is not None else np.zeros(1),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        expl_mu=mu, expl_sd=sd,
    )

    print(f"[ok] taste_model.npz")
    print(f"     Bradley-Terry:  {n_pairs} coppie · accuracy sul training {acc_bt:.1%}")
    print(f"     controllo binario: accuracy sul training {acc_bin:.1%}")
    print()
    print("     ⚠️  L'accuracy sul TRAINING non significa niente (può essere overfitting).")
    print("     Il numero che conta è quello di step7 sul GIORNO B.")

    # interpretabilità: quali feature esplicite pesano?
    labels = ["nitidezza", "esposizione", "contrasto", "n_volti", "area_volti"]
    print("\n     Pesi appresi (feature esplicite):")
    for lab, w in zip(labels, w_bt[:5]):
        bar = "█" * int(abs(w) * 8)
        sign = "+" if w >= 0 else "−"
        print(f"       {lab:<12} {sign}{abs(w):.2f} {bar}")
    print(f"       (+ {len(w_bt) - 5} dim. di embedding PCA)")


if __name__ == "__main__":
    main()
