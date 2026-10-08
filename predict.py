"""
Réutilise le modèle entraîné (har_lstm.pt) SANS réentraîner.

Exemples :
    python predict.py                      # démo sur quelques fenêtres du jeu de test
    python predict.py --eval               # précision complète sur le jeu de test
    python predict.py --fichier mesure.npy # vos propres capteurs

Format attendu pour --fichier : un .npy ou .csv de forme (128, 9) ou (n, 128, 9)
  128 pas de temps (2,56 s à 50 Hz), 9 canaux dans cet ordre :
  body_acc x,y,z | body_gyro x,y,z | total_acc x,y,z
  (accélération en g, gyroscope en rad/s, comme dans UCI HAR)
"""
import argparse

import numpy as np

from har_common import charger, charger_modele, get_device, predire, telecharger

parser = argparse.ArgumentParser()
parser.add_argument("--modele", default="har_lstm.pt")
parser.add_argument("--fichier", help="fichier .npy ou .csv de mesures")
parser.add_argument("--eval", action="store_true", help="évaluer sur le jeu de test")
args = parser.parse_args()

device = get_device()
modele, moy, std, activites = charger_modele(args.modele, device)
print(f"Modèle chargé depuis {args.modele} (appareil : {device})")

if args.fichier:
    if args.fichier.endswith(".npy"):
        X = np.load(args.fichier)
    else:
        X = np.loadtxt(args.fichier, delimiter=",")
    X = X.reshape(-1, 128, 9)
    proba = predire(modele, moy, std, X, device)
    for i, p in enumerate(proba):
        k = p.argmax()
        print(f"Fenêtre {i}: {activites[k]} ({p[k]*100:.1f} %)")

else:
    telecharger()
    X_test, y_test, _ = charger("test")
    proba = predire(modele, moy, std, X_test, device)
    y_pred = proba.argmax(1)

    if args.eval:
        print(f"Précision sur tout le test : {(y_pred == y_test).mean():.3f}")
    else:
        rng = np.random.default_rng(0)
        for i in rng.choice(len(X_test), 8, replace=False):
            k = y_pred[i]
            ok = "OK " if k == y_test[i] else "ERR"
            print(f"[{ok}] réel : {activites[y_test[i]]:<20} "
                  f"prédit : {activites[k]:<20} ({proba[i, k]*100:.1f} %)")
