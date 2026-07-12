"""
Predittore di estetica "oggettiva" LAION (sa_0_4_vit_b_32_linear.pth) — un layer
lineare addestrato su giudizi estetici umani aggregati (AVA + SAC + LAION-Logos),
sopra gli stessi embedding CLIP ViT-B/32 già calcolati in step1. Zero calcolo extra
sulle foto: si applica direttamente a features.npz.

Non è "il gusto di questa utente" — è un consenso estetico generico. Serve come
segnale indipendente da combinare col taste centroid, non per sostituirlo.
"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WEIGHTS_URL = "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_b_32_linear.pth"
WEIGHTS_PATH = Path(__file__).parent / "aesthetic_vit_b_32.pth"


def _ensure_weights():
    if not WEIGHTS_PATH.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


def load_aesthetic_model():
    sd = torch.load(_ensure_weights(), map_location="cpu", weights_only=True)
    model = nn.Linear(512, 1)
    model.load_state_dict(sd)
    model.eval()
    return model


def aesthetic_scores(embs):
    """embs: (N, 512) già normalizzati (norma 1), come in features.npz. Ritorna score grezzi (~1-10)."""
    model = load_aesthetic_model()
    with torch.no_grad():
        x = torch.from_numpy(embs.astype(np.float32))
        return model(x).squeeze(-1).numpy()
