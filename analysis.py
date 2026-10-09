"""End to end: fit archetypes, validate them, build the similarity engine, test team impact.

Run from the repo root:  python src/analysis.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import normalize

from cluster import (DATA_MIN_MP, MIN_MP, MODEL_FEATURES, OUT, SEED, add_position_index,
                     embed, fit_space)
from features import DATA, player_seasons, player_team_seasons

from scipy.optimize import linear_sum_assignment

K = 11
# Each archetype name has a prototype profile written from first inspection of the clusters.
# Clusters get names by optimal one to one matching against these prototypes, so labels
# survive refits that reshuffle cluster ids.
PROTO_COLS = ["pts100", "ast100", "fg3a100", "orb100", "drb100", "blk100", "stl100",
              "rim_share", "mid_share", "three_share", "corner3_share", "pos_index"]
PROTOTYPES = {
    "Post Scoring Big":  [20.2, 2.8, 0.1, 4.7, 9.6, 1.9, 1.3, .43, .14, .00, .04, 4.6],
    "Two Way Wing":      [16.2, 3.6, 5.0, 2.1, 5.9, 0.9, 2.1, .33, .06, .38, .45, 2.8],
    "Stretch Forward":   [22.7, 3.6, 7.0, 2.4, 8.0, 1.2, 1.3, .27, .08, .39, .25, 3.9],
    "Midrange Scorer":   [22.6, 3.5, 2.0, 2.5, 7.5, 1.1, 1.4, .26, .18, .10, .36, 3.7],
    "Primary Creator":   [32.8, 8.0, 8.7, 1.2, 6.0, 0.6, 1.7, .22, .13, .35, .13, 2.0],
    "Floor General":     [20.9, 9.1, 5.5, 1.0, 4.8, 0.4, 2.0, .25, .11, .31, .20, 1.3],
    "Rim Running Big":   [18.2, 2.4, 0.1, 5.7, 10.7, 2.6, 1.3, .68, .04, .01, .07, 4.9],
    "Hub Big":           [30.6, 5.2, 3.5, 3.8, 11.0, 1.8, 1.5, .34, .12, .15, .15, 4.6],
    "Secondary Scorer":  [22.4, 4.6, 8.6, 1.0, 4.8, 0.5, 1.5, .17, .10, .46, .23, 2.2],
    "Slashing Forward":  [19.8, 3.2, 1.5, 4.6, 9.2, 1.9, 1.5, .50, .07, .10, .45, 4.4],
    "Spot Up Shooter":   [17.5, 3.1, 9.2, 1.2, 5.4, 0.7, 1.4, .15, .05, .64, .31, 2.9],
}


def name_clusters(train):
    prof = train.groupby("archetype_id")[PROTO_COLS].mean()
    sd = train[PROTO_COLS].std()
    P = pd.DataFrame(PROTOTYPES, index=PROTO_COLS).T
    cost = np.array([[(((prof.loc[c] - P.loc[n]) / sd) ** 2).sum() for n in P.index]
                     for c in prof.index])
    rows, cols = linear_sum_assignment(cost)
    worst = cost[rows, cols].max()
    assert worst < 3.0, f"a cluster no longer matches any prototype (cost {worst:.2f})"
    return {int(prof.index[r]): P.index[c] for r, c in zip(rows, cols)}


def fit_archetypes():
    allp = add_position_index(player_seasons())
    train = allp[allp.mp >= MIN_MP].reset_index(drop=True)
    scaler, pca = fit_space(train)
    X = embed(train, scaler, pca)
    km = KMeans(K, n_init=50, random_state=SEED).fit(X)
    train["archetype_id"] = km.labels_
    labels = name_clusters(train)
    train["archetype"] = train.archetype_id.map(labels)
    ward = AgglomerativeClustering(K, linkage="ward").fit_predict(X)
    ari_ward = adjusted_rand_score(km.labels_, ward)

    # Assign every player season with real minutes to its nearest archetype.
    pool = allp[allp.mp >= DATA_MIN_MP].reset_index(drop=True)
    Xp = embed(pool, scaler, pca)
    pool["archetype_id"] = km.predict(Xp)
    pool["archetype"] = pool.archetype_id.map(labels)
    return train, pool, Xp, scaler, pca, km, ari_ward


def profiles(train):
    prof = train.groupby("archetype")[MODEL_FEATURES].mean()
    prof["player_seasons"] = train.archetype.value_counts()
    return prof


def similarity_index(pool, Xp):
    return normalize(Xp)


def most_similar(pool, V, player, season, n=5, other_players_only=True):
    hit = pool[(pool.player == player) & (pool.season == season)]
    if hit.empty:
        raise KeyError(f"{player} {season} not found")
    i = hit.index[0]
    sims = V @ V[i]
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == i or (other_players_only and pool.player_id[j] == pool.player_id[i]):
            continue
        out.append((pool.player[j], int(pool.season[j]), pool.archetype[j], round(float(sims[j]), 3)))
        if len(out) == n:
            break
    return out


def similarity_validation(pool, V):
    """For each player season, where does that same player's next season rank among all others?

    A good similarity space puts a player's next season near the top, since roles change slowly.
    """
    key = {(r.player_id, r.season): i for i, r in pool.iterrows()}
    pct = []
    for (pid, s), i in key.items():
        j = key.get((pid, s + 1))
        if j is None:
            continue
        sims = V @ V[i]
        sims[i] = -np.inf
        rank = (sims > sims[j]).sum()  # 0 means the closest of all
        pct.append(rank / (len(V) - 1))
    pct = np.array(pct)
    return {
        "pairs": int(len(pct)),
        "median_percentile": float(np.median(pct)),
        "share_in_top_1pct": float((pct <= 0.01).mean()),
        "share_in_top_5pct": float((pct <= 0.05).mean()),
    }


def evolution(pool):
    m = pool.groupby(["season", "archetype"]).mp.sum().unstack(fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


TEAM_MAP = {"BKN": "BRK", "PHX": "PHO"}


def team_results():
    t = pd.read_csv(DATA / "regular_season_totals_2010_2024.csv")
    t["season"] = t.SEASON_YEAR.str[:4].astype(int) + 1
    t["team"] = t.TEAM_ABBREVIATION.replace(TEAM_MAP)
    t.loc[(t.team == "CHA") & (t.season >= 2015), "team"] = "CHO"
    g = t.groupby(["season", "team"]).agg(games=("GAME_ID", "nunique"),
                                          point_diff=("PLUS_MINUS", "mean"),
                                          win_pct=("WL", lambda s: (s == "W").mean()))
    return g.reset_index()


def team_analysis(pool):
    stints = player_team_seasons()
    stints = stints[~stints.team.str.match(r"\dTM")][["season", "player_id", "team", "mp"]]
    role = pool[["season", "player_id", "archetype"]]
    st = stints.merge(role, on=["season", "player_id"], how="inner")

    # Talent control: each player's PRIOR season RAPTOR. Same season RAPTOR is team adjusted
    # (player values are tuned to add up to the team's results), so using it here would leak the
    # answer into the predictor. Prior ratings are shrunk toward replacement level (-2) by
    # 500 minutes, and players with no prior season (mostly rookies) get replacement level.
    REPLACEMENT, SHRINK_MP = -2.0, 500
    raptor = pd.read_csv(DATA / "modern_RAPTOR_by_player.csv")[["player_id", "season", "raptor_total", "mp"]]
    prior = raptor.assign(season=raptor.season + 1).rename(columns={"raptor_total": "r_prev", "mp": "mp_prev"})
    st = st.merge(prior, on=["season", "player_id"], how="left")
    st["prior_rating"] = ((st.r_prev * st.mp_prev + REPLACEMENT * SHRINK_MP) /
                          (st.mp_prev + SHRINK_MP)).fillna(REPLACEMENT)

    shares = st.pivot_table(index=["season", "team"], columns="archetype", values="mp",
                            aggfunc="sum", fill_value=0)
    covered = shares.sum(axis=1)
    shares = shares.div(covered, axis=0)

    talent = (st.assign(w=st.prior_rating * st.mp)
                .groupby(["season", "team"])
                .apply(lambda d: d.w.sum() / d.mp.sum(), include_groups=False)
                .rename("talent"))
    # RAPTOR covers 2014 to 2022, so prior season ratings exist for 2015 to 2022.
    df = (shares.join(talent).reset_index()
                .merge(team_results(), on=["season", "team"], how="inner"))
    df = df[df.season.between(2015, 2022)].reset_index(drop=True)

    arch = [c for c in PROTOTYPES if c in df.columns]
    y = df.point_diff.to_numpy()
    groups = df.season.to_numpy()
    cv = GroupKFold(n_splits=df.season.nunique())

    def cv_r2(cols):
        Xm = df[cols].to_numpy()
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 25)), Xm, y, cv=cv, groups=groups)
        return float(r2_score(y, pred))

    res = {
        "team_seasons": int(len(df)),
        "seasons": [int(df.season.min()), int(df.season.max())],
        "r2_talent_only": cv_r2(["talent"]),
        "r2_archetypes_only": cv_r2(arch[:-1]),  # drop one share, they sum to 1
        "r2_talent_plus_archetypes": cv_r2(["talent"] + arch[:-1]),
    }

    # Which archetypes help after holding talent fixed? Plain OLS for readable coefficients.
    import statsmodels.api as sm
    base = "Spot Up Shooter"
    cols = ["talent"] + [a for a in arch if a != base]
    Xo = sm.add_constant(df[cols] * np.r_[1, np.full(len(cols) - 1, 10)])  # per 10 pts of minute share
    ols = sm.OLS(y, Xo).fit(cov_type="cluster", cov_kwds={"groups": groups})
    coef = pd.DataFrame({"coef": ols.params, "p": ols.pvalues}).drop("const")
    res["ols_talent_coef"] = float(ols.params["talent"])
    res["ols_baseline"] = base
    return df, res, coef


def main():
    train, pool, Xp, scaler, pca, km, ari_ward = fit_archetypes()
    prof = profiles(train)
    prof.round(3).to_csv(OUT / "archetype_profiles.csv")

    V = similarity_index(pool, Xp)
    val = similarity_validation(pool, V)
    evo = evolution(pool)
    evo.to_csv(OUT / "archetype_minute_share_by_season.csv")
    pool[["season", "player", "player_id", "team", "mp", "archetype"]].to_csv(
        OUT / "player_archetypes.csv", index=False)

    df, team_res, coef = team_analysis(pool)
    df.to_csv(OUT / "team_seasons.csv", index=False)
    coef.round(4).to_csv(OUT / "team_ols_coefficients.csv")

    examples = {}
    for p, s in [("Nikola Jokić", 2024), ("Stephen Curry", 2016), ("Giannis Antetokounmpo", 2020),
                 ("Draymond Green", 2016), ("Rudy Gobert", 2022), ("Victor Wembanyama", 2026)]:
        try:
            examples[f"{p} {s}"] = most_similar(pool, V, p, s)
        except KeyError as e:
            examples[f"{p} {s}"] = str(e)

    results = {
        "train_player_seasons": int(len(train)),
        "pool_player_seasons": int(len(pool)),
        "seasons": [int(pool.season.min()), int(pool.season.max())],
        "features": len(MODEL_FEATURES),
        "pca_components": int(pca.n_components_),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "k": K,
        "ward_agreement_ari": float(ari_ward),
        "similarity_validation": val,
        "share_change": {a: [float(evo.loc[evo.index.min(), a]), float(evo.loc[evo.index.max(), a])]
                         for a in evo.columns},
        "team": team_res,
        "examples": examples,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(coef.round(3).to_string())


if __name__ == "__main__":
    main()
