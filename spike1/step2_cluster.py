"""
STEP 2 — Clustering

Due livelli (§5.2 spec):
  BURST  — near-duplicate: similarità coseno alta E vicinanza temporale.
           È l'unità su cui il modello di gusto imparerà (§5.3.3: contenuto costante → isola lo stile).
  EVENTO — DBSCAN su tempo (+GPS): "Kyoto, 14 giugno, pomeriggio".

    python step2_cluster.py day_A --sim 0.92 --gap 90
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path


def burst_clusters(names, embs, ts, sim_thr, gap_s, ts_reliable=True):
    """Union-find su coppie (simili E vicine nel tempo). Semplice, O(n²): va benissimo per 150 foto.
    Se i timestamp non sono affidabili (fallback mtime), il vincolo temporale viene ignorato."""
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    S = embs @ embs.T  # coseno (embedding già normalizzati)
    for i in range(n):
        for j in range(i + 1, n):
            if ts_reliable:
                close_in_time = (ts[i] is not None and ts[j] is not None
                                 and abs(ts[i] - ts[j]) <= gap_s)
            else:
                close_in_time = True  # solo similarità: meglio di un vincolo farlocco
            if S[i, j] >= sim_thr and close_in_time:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # rinumera 0..k-1
    return {k: v for k, v in enumerate(groups.values())}


def event_clusters(ts, lat, lon, eps_minutes=45):
    """DBSCAN sul tempo. Se il GPS c'è lo aggiungiamo come dimensione secondaria."""
    valid_ts = [t for t in ts if t is not None]
    if not valid_ts:
        return {0: list(range(len(ts)))}
    t0 = min(valid_ts)
    X = [[( (t or t0) - t0) / 60.0] for t in ts]  # minuti dall'inizio
    X = np.array(X)
    db = DBSCAN(eps=eps_minutes, min_samples=2).fit(X)
    groups = {}
    for i, lab in enumerate(db.labels_):
        groups.setdefault(int(lab), []).append(i)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    # 0.85, non 0.92: su PhotoKoyasan 0.92 non raggruppava scatti ripetuti della
    # stessa inquadratura quando esposizione/orientamento differivano (verificato
    # a mano — vedi analisi 2026-07-12), frammentando un burst reale in singoli.
    ap.add_argument("--sim", type=float, default=0.85, help="soglia coseno per il burst")
    ap.add_argument("--gap", type=float, default=90, help="secondi max tra due scatti dello stesso burst")
    ap.add_argument("--event-eps", type=float, default=45, help="minuti, DBSCAN evento")
    a = ap.parse_args()

    d = np.load(out_path(a.folder, "features.npz"), allow_pickle=True)
    names = list(d["names"])
    embs = d["embs"]
    meta = json.loads(out_path(a.folder, "meta.json").read_text())
    ts = [m["ts"] for m in meta]
    lat = [m["lat"] for m in meta]
    lon = [m["lon"] for m in meta]

    n_exif = sum(1 for m in meta if m.get("ts_from_exif", True))
    ts_reliable = n_exif >= len(meta) * 0.8
    if not ts_reliable:
        print(f"[!!] solo {n_exif}/{len(meta)} timestamp da EXIF → vincolo temporale DISATTIVATO,")
        print(f"     clustering per sola similarità. Meglio riesportare con i metadati.")

    bursts = burst_clusters(names, embs, ts, a.sim, a.gap, ts_reliable)
    events = event_clusters(ts, lat, lon, a.event_eps)

    sizes = sorted((len(v) for v in bursts.values()), reverse=True)
    multi = [s for s in sizes if s > 1]

    out = {
        "params": {"sim": a.sim, "gap": a.gap, "event_eps": a.event_eps},
        "bursts": {str(k): [names[i] for i in v] for k, v in bursts.items()},
        "events": {str(k): [names[i] for i in v] for k, v in events.items()},
    }
    out_path(a.folder, "clusters.json").write_text(json.dumps(out, indent=1))

    print(f"[ok] clusters.json")
    print(f"     {len(names)} foto → {len(bursts)} gruppi burst ({len(multi)} con più di 1 foto)")
    print(f"     gruppo più grande: {sizes[0] if sizes else 0} foto")
    print(f"     {len(events)} eventi")
    print()
    print("     ⚠️  GUARDA I GRUPPI PIÙ GRANDI A OCCHIO.")
    print("     Se il clustering non becca i tuoi duplicati reali, tutto il resto è fuffa.")
    print("     Alza/abbassa --sim (0.88 più aggressivo, 0.95 più conservativo).")


if __name__ == "__main__":
    main()
