"""
STEP 1 — Feature

Per ogni foto:
  - embedding CLIP (contenuto)     → serve a clustering, unicità, diversità
  - nitidezza (laplacian variance) → qualità tecnica
  - esposizione                    → qualità tecnica
  - volti (n, area)                → §4.3 spec: le persone battono i tramonti
  - EXIF (timestamp, GPS)          → clustering per evento

    python step1_features.py day_A
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import list_images, load_image, read_exif, out_path

MODEL = "clip-ViT-B-32"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def quality_features(pil_img):
    """Nitidezza + esposizione. Su immagine ridotta: ci interessa il relativo, non l'assoluto."""
    img = np.array(pil_img.convert("L").resize((512, 512)))
    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    mean = float(img.mean()) / 255.0
    # penalizza sotto/sovraesposizione: 1.0 a metà istogramma, 0 agli estremi
    exposure = 1.0 - abs(mean - 0.5) * 2.0
    contrast = float(img.std()) / 128.0
    return sharpness, exposure, contrast


def face_features(pil_img, cascade):
    img = np.array(pil_img.convert("L"))
    h, w = img.shape
    faces = cascade.detectMultiScale(img, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return 0, 0.0
    area = sum(fw * fh for (_, _, fw, fh) in faces) / float(w * h)
    return int(len(faces)), float(area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()

    files = list_images(a.folder)
    if not files:
        raise SystemExit(f"[stop] nessuna immagine in {a.folder}/")
    print(f"[i] {len(files)} foto")

    from sentence_transformers import SentenceTransformer
    print(f"[i] carico {MODEL} …")
    model = SentenceTransformer(MODEL)
    cascade = cv2.CascadeClassifier(FACE_CASCADE)

    names, embs, rows = [], [], []
    for f in tqdm(files, desc="feature"):
        img = load_image(f, max_side=512)
        emb = model.encode(img, convert_to_numpy=True, show_progress_bar=False)
        sharp, expo, contr = quality_features(img)
        n_faces, face_area = face_features(img, cascade)
        ts, lat, lon, from_exif = read_exif(f)

        names.append(f.name)
        embs.append(emb)
        rows.append(dict(
            name=f.name, sharpness=sharp, exposure=expo, contrast=contr,
            n_faces=n_faces, face_area=face_area,
            ts=ts, lat=lat, lon=lon, ts_from_exif=from_exif,
        ))

    embs = np.vstack(embs).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)  # normalizzati → dot = coseno

    np.savez(out_path(a.folder, "features.npz"), names=np.array(names), embs=embs)
    out_path(a.folder, "meta.json").write_text(json.dumps(rows, indent=1))

    n_with_faces = sum(1 for r in rows if r["n_faces"] > 0)
    n_with_gps = sum(1 for r in rows if r["lat"] is not None)
    n_exif_ts = sum(1 for r in rows if r["ts_from_exif"])
    print(f"[ok] features.npz + meta.json")
    print(f"     volti rilevati in {n_with_faces}/{len(rows)} foto · GPS in {n_with_gps}/{len(rows)}")
    print(f"     timestamp da EXIF: {n_exif_ts}/{len(rows)}")
    if n_exif_ts < len(rows) * 0.8:
        print("     [!!] MOLTI TIMESTAMP NON VENGONO DALL'EXIF (fallback su mtime).")
        print("          Dopo un export gli mtime sono tutti uguali → il clustering temporale")
        print("          diventa spazzatura. Riesporta da Foto CON i metadati (opzione")
        print("          'Includi info posizione' + formato originale o JPEG con EXIF).")
    if n_with_gps == 0:
        print("     [!] nessun GPS: il clustering per evento userà solo il tempo")


if __name__ == "__main__":
    main()
