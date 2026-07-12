"""Utility condivise dallo spike. Codice usa-e-getta: nessun error handling serio."""
import json
import os
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

# iPhone esporta HEIC: registriamo il decoder
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("[warn] pillow-heif non installato: gli HEIC non si apriranno")

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def list_images(folder):
    """Tutte le immagini della cartella, ordinate per nome.
    Esclude lambda_sweep.png: è un output di step4 (grafico), non una foto —
    senza questo filtro un secondo giro di step0/step1 sulla stessa cartella
    lo riprocesserebbe come se fosse la 134esima foto del giorno."""
    p = Path(folder)
    return sorted([f for f in p.iterdir()
                   if f.suffix.lower() in IMG_EXT and f.name != "lambda_sweep.png"])


def load_image(path, max_side=None):
    img = Image.open(path).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _dms_to_deg(dms, ref):
    try:
        d, m, s = [float(x) for x in dms]
        val = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


def read_exif(path):
    """Restituisce (timestamp_unix, lat | None, lon | None, from_exif: bool)."""
    ts = lat = lon = None
    try:
        exif = Image.open(path).getexif()
    except Exception:
        exif = None

    if exif:
        # DateTimeOriginal (36867) vive nell'Exif IFD (0x8769): è lì che lo scrive l'iPhone.
        # DateTime (306) nel base IFD è il fallback.
        raw = None
        try:
            raw = exif.get_ifd(0x8769).get(36867)
        except Exception:
            pass
        raw = raw or exif.get(306)
        if raw:
            try:
                ts = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").timestamp()
            except ValueError:
                pass

        try:
            gps = exif.get_ifd(_EXIF_TAGS.get("GPSInfo")) or {}
            lat = _dms_to_deg(gps.get(_GPS_TAGS["GPSLatitude"]), gps.get(_GPS_TAGS["GPSLatitudeRef"]))
            lon = _dms_to_deg(gps.get(_GPS_TAGS["GPSLongitude"]), gps.get(_GPS_TAGS["GPSLongitudeRef"]))
        except Exception:
            pass

    from_exif = ts is not None
    if ts is None:
        # fallback: data di modifica. ⚠️ dopo un export gli mtime sono tutti uguali:
        # il chiamante deve sapere che il timestamp NON è affidabile.
        ts = os.path.getmtime(path)

    return ts, lat, lon, from_exif


def load_favorites(folder):
    """favorites.txt opzionale: un nome file per riga."""
    f = Path(folder) / "favorites.txt"
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}


def load_ground_truth(folder):
    f = Path(folder) / "ground_truth.json"
    if not f.exists():
        raise SystemExit(
            f"[stop] Manca {f}\n"
            "       Il rituale del sabato mattina non è negoziabile:\n"
            "       python step0_picker.py <cartella>  →  scegli le tue 15  →  salva il JSON."
        )
    data = json.loads(f.read_text())
    return set(data["selected"])


def out_path(folder, name):
    return Path(folder) / name


def normalize(x):
    """Min-max su array numpy, robusto al caso costante."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
