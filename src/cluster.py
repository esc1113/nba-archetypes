"""Fit player archetypes with KMeans, check them with Ward hierarchical clustering."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FEATURES, player_seasons

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)
MIN_MP = 1000      # minutes to help define archetypes
DATA_MIN_MP = 250   # minutes to be assigned an archetype
SEED = 7


def add_position_index(df):
    """Collapse five position shares into one 1 (PG) to 5 (C) number."""
    w = np.array([1, 2, 3, 4, 5])
    shares = df[["pg_pct", "sg_pct", "sf_pct", "pf_pct", "c_pct"]].to_numpy()
    df = df.copy()
    df["pos_index"] = (shares * w).sum(1) / shares.sum(1).clip(min=1)
    return df


MODEL_FEATURES = [f for f in FEATURES if not f.endswith("_pct")] + ["pos_index"]


def fit_space(train):
    scaler = StandardScaler().fit(train[MODEL_FEATURES])
    z = scaler.transform(train[MODEL_FEATURES])
    pca = PCA(n_components=0.90, random_state=SEED).fit(z)
    return scaler, pca


def embed(df, scaler, pca):
    return pca.transform(scaler.transform(df[MODEL_FEATURES]))


def stability(X, k, reps=20):
    """Mean ARI between a model fit on an 80% subsample and the full fit."""
    rng = np.random.default_rng(SEED)
    full = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
    scores = []
    for r in range(reps):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        km = KMeans(k, n_init=10, random_state=r).fit(X[idx])
        scores.append(adjusted_rand_score(full, km.predict(X)))
    return float(np.mean(scores))


def choose_k(X, ks=range(4, 15)):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_, random_state=SEED),
            "stability_ari": stability(X, k),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allp = add_position_index(player_seasons())
    train = allp[allp.mp >= MIN_MP].reset_index(drop=True)
    scaler, pca = fit_space(train)
    X = embed(train, scaler, pca)
    print("PCA components kept:", pca.n_components_,
          "variance:", pca.explained_variance_ratio_.sum().round(3))
    sel = choose_k(X)
    sel.to_csv(OUT / "k_selection.csv", index=False)
    print(sel.round(3).to_string(index=False))
