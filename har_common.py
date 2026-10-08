"""Code partagé entre train.py et predict.py (données, modèle, chargement)."""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
       "UCI%20HAR%20Dataset.zip")
# Les données sont stockées dans le dossier "data" à côté de ce fichier,
# donc toujours au même endroit, quel que soit le dossier d'où on lance Python.
DATA_ROOT = Path(__file__).resolve().parent / "data"
DATA_DIR = DATA_ROOT / "UCI HAR Dataset"
SIGNAUX = ["body_acc", "body_gyro", "total_acc"]  # x, y, z chacun -> 9 canaux
ACTIVITES = ["Marche", "Montee escaliers", "Descente escaliers",
             "Assis", "Debout", "Allonge"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ données
def _cache(split):
    return DATA_ROOT / f"cache_{split}.npz"


def telecharger():
    """Télécharge le jeu de données UNE seule fois. Ne fait rien s'il est déjà là."""
    fichiers_ok = all((DATA_DIR / s / f"y_{s}.txt").exists() for s in ("train", "test"))
    cache_ok = all(_cache(s).exists() for s in ("train", "test"))
    if fichiers_ok or cache_ok:
        return

    DATA_ROOT.mkdir(exist_ok=True)
    archive = DATA_ROOT / "har.zip"
    print("Téléchargement du jeu de données UCI HAR (une seule fois)...")
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA_ROOT)
    archive.unlink()
    print(f"Données enregistrées dans : {DATA_ROOT}")


def charger(split):
    """split = 'train' ou 'test'. Retourne X (n,128,9), y (n,), sujets (n,).
    La 1re fois, lit les fichiers texte (lent) puis crée un cache .npz rapide."""
    if _cache(split).exists():
        d = np.load(_cache(split))
        return d["X"], d["y"], d["sujets"]

    X, y, sujets = _lire_fichiers_texte(split)
    DATA_ROOT.mkdir(exist_ok=True)
    np.savez_compressed(_cache(split), X=X, y=y, sujets=sujets)
    return X, y, sujets


def _lire_fichiers_texte(split):
    canaux = []
    for s in SIGNAUX:
        for axe in "xyz":
            chemin = f"{DATA_DIR}/{split}/Inertial Signals/{s}_{axe}_{split}.txt"
            canaux.append(np.loadtxt(chemin))
    X = np.stack(canaux, axis=-1).astype(np.float32)
    y = np.loadtxt(f"{DATA_DIR}/{split}/y_{split}.txt").astype(np.int64) - 1
    sujets = np.loadtxt(f"{DATA_DIR}/{split}/subject_{split}.txt").astype(int)
    return X, y, sujets


# ------------------------------------------------------------------ modèle
class LSTMHAR(nn.Module):
    def __init__(self, n_canaux=9, hidden=64, n_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_canaux, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.tete = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, n_classes),
        )

    def forward(self, x):                # x : (batch, 128, 9)
        sortie, _ = self.lstm(x)
        return self.tete(sortie[:, -1])  # dernier pas de temps -> logits


# ------------------------------------------------------------------ sauvegarde / chargement
def sauvegarder(chemin, modele, moy, std, hidden=64):
    """Sauvegarde les poids + la normalisation (indispensable pour réutiliser)."""
    torch.save({
        "etat_modele": modele.state_dict(),
        "moy": torch.tensor(moy),
        "std": torch.tensor(std),
        "hidden": hidden,
        "activites": ACTIVITES,
    }, chemin)


def charger_modele(chemin, device=None):
    device = device or get_device()
    ckpt = torch.load(chemin, map_location=device, weights_only=True)
    modele = LSTMHAR(hidden=ckpt["hidden"]).to(device)
    modele.load_state_dict(ckpt["etat_modele"])
    modele.eval()
    return modele, ckpt["moy"].numpy(), ckpt["std"].numpy(), ckpt["activites"]


@torch.no_grad()
def predire(modele, moy, std, X, device=None):
    """X : (n,128,9) ou (128,9). Retourne les probabilités (n, 6)."""
    device = device or get_device()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    X = (X - moy) / std
    logits = modele(torch.from_numpy(X).to(device))
    return torch.softmax(logits, dim=1).cpu().numpy()
