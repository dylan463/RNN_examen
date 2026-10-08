"""
Entraîne le LSTM sur UCI HAR, évalue sur le test et sauvegarde le modèle.

    python train.py
Produit : har_lstm.pt, courbes.png, confusion.png, seuil_alerte.png
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from har_common import (ACTIVITES, LSTMHAR, charger, get_device, predire,
                        sauvegarder, telecharger)

SEED = 42
EPOCHS = 40
BATCH = 64
LR = 1e-3
PATIENCE = 6
HIDDEN = 64
CHEMIN_MODELE = "har_lstm.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = get_device()
print("Appareil utilisé :", device)

# ------------------------------------------------------------------ données
telecharger()
X_all, y_all, sujets = charger("train")
X_test, y_test, _ = charger("test")

# Validation séparée par sujet (une personne n'est jamais dans train ET val)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
i_tr, i_val = next(gss.split(X_all, y_all, groups=sujets))
X_train, y_train = X_all[i_tr], y_all[i_tr]
X_val, y_val = X_all[i_val], y_all[i_val]

# Normalisation par canal, calculée sur le train uniquement
moy = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
norm = lambda a: (a - moy) / std


def loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(norm(X)), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


dl_train, dl_val = loader(X_train, y_train, True), loader(X_val, y_val, False)

# ------------------------------------------------------------------ entraînement
modele = LSTMHAR(hidden=HIDDEN).to(device)
critere = nn.CrossEntropyLoss()
optim = torch.optim.Adam(modele.parameters(), lr=LR)


def epoque(dl, entrainer):
    modele.train(entrainer)
    perte, bons, n = 0.0, 0, 0
    with torch.set_grad_enabled(entrainer):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = modele(xb)
            loss = critere(logits, yb)
            if entrainer:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                optim.step()
            perte += loss.item() * len(yb)
            bons += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return perte / n, bons / n


histo = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
meilleure_val, sans_progres = float("inf"), 0

for ep in range(1, EPOCHS + 1):
    l, a = epoque(dl_train, True)
    vl, va = epoque(dl_val, False)
    for k, v in zip(histo, (l, vl, a, va)):
        histo[k].append(v)
    print(f"Epoque {ep:02d} | perte {l:.4f} acc {a:.3f} | val perte {vl:.4f} acc {va:.3f}")

    if vl < meilleure_val:                      # on garde le meilleur modèle
        meilleure_val, sans_progres = vl, 0
        sauvegarder(CHEMIN_MODELE, modele, moy, std, HIDDEN)
    else:
        sans_progres += 1
        if sans_progres >= PATIENCE:
            print("Arrêt anticipé.")
            break

# On recharge le meilleur modèle sauvegardé pour l'évaluation finale
ckpt = torch.load(CHEMIN_MODELE, map_location=device, weights_only=True)
modele.load_state_dict(ckpt["etat_modele"])
modele.eval()
print(f"\nMeilleur modèle sauvegardé dans {CHEMIN_MODELE}")

# ------------------------------------------------------------------ évaluation
proba = predire(modele, moy, std, X_test, device)
y_pred = proba.argmax(1)
print(f"Précision sur le test : {(y_pred == y_test).mean():.3f}\n")
print(classification_report(y_test, y_pred, target_names=ACTIVITES, digits=3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(histo["loss"], label="train"); ax[0].plot(histo["val_loss"], label="validation")
ax[0].set_title("Perte"); ax[0].legend()
ax[1].plot(histo["acc"], label="train"); ax[1].plot(histo["val_acc"], label="validation")
ax[1].set_title("Précision"); ax[1].legend()
plt.tight_layout(); plt.savefig("courbes.png", dpi=150)

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0],
            xticklabels=ACTIVITES, yticklabels=ACTIVITES)
ax[0].set_title("Matrice de confusion (effectifs)")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax[1],
            xticklabels=ACTIVITES, yticklabels=ACTIVITES)
ax[1].set_title("Matrice de confusion (normalisée)")
for a in ax:
    a.set_xlabel("Prédit"); a.set_ylabel("Réel"); a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("confusion.png", dpi=150)

print("Confusions les plus fréquentes :")
paires = [(ACTIVITES[i], ACTIVITES[j], cm[i, j])
          for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
for reel, pred, n in sorted(paires, key=lambda t: -t[2])[:5]:
    print(f"  {reel} -> {pred} : {n}")


# ------------------------------------------------------------------ fausses alertes
def analyser_seuil(y_binaire, proba_classe):
    """Compromis fausses alertes / détections ratées.
    A utiliser avec de vraies chutes (SisFall, MobiAct) : 1 = chute, 0 = normal."""
    prec, rap, seuils = precision_recall_curve(y_binaire, proba_classe)
    plt.figure(figsize=(6, 4))
    plt.plot(seuils, prec[:-1], label="Précision (peu de fausses alertes)")
    plt.plot(seuils, rap[:-1], label="Rappel (événements détectés)")
    plt.xlabel("Seuil de décision"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("seuil_alerte.png", dpi=150)


# Démonstration sur UCI HAR (pas de vraies chutes) avec la classe "Allonge"
analyser_seuil((y_test == 5).astype(int), proba[:, 5])
print("\nTerminé. Fichiers : har_lstm.pt, courbes.png, confusion.png, seuil_alerte.png")
