"""Build one row per player season with role features (how a player plays, not how well)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
KEYS = ["season", "player_id", "player", "age", "team", "pos", "g", "gs", "mp"]

# Role features. Efficiency stats (TS%, FG%) are left out on purpose so clusters
# describe what a player does, and quality can be tested separately.
PER100 = {
    "pts_per_100_poss": "pts100",
    "fga_per_100_poss": "fga100",
    "x3pa_per_100_poss": "fg3a100",
    "fta_per_100_poss": "fta100",
    "orb_per_100_poss": "orb100",
    "drb_per_100_poss": "drb100",
    "ast_per_100_poss": "ast100",
    "stl_per_100_poss": "stl100",
    "blk_per_100_poss": "blk100",
    "tov_per_100_poss": "tov100",
    "pf_per_100_poss": "pf100",
}
SHOOT = {
    "avg_dist_fga": "shot_dist",
    "percent_fga_from_x0_3_range": "rim_share",
    "percent_fga_from_x3_10_range": "short_share",
    "percent_fga_from_x10_16_range": "mid_share",
    "percent_fga_from_x16_3p_range": "long2_share",
    "percent_fga_from_x3p_range": "three_share",
    "percent_assisted_x2p_fg": "ast2_rate",
    "percent_assisted_x3p_fg": "ast3_rate",
    "percent_dunks_of_fga": "dunk_share",
    "percent_corner_3s_of_3pa": "corner3_share",
}
PBP = {
    "pg_percent": "pg_pct",
    "sg_percent": "sg_pct",
    "sf_percent": "sf_pct",
    "pf_percent": "pf_pct",
    "c_percent": "c_pct",
}
FEATURES = list(PER100.values()) + list(SHOOT.values()) + list(PBP.values())


def _load(name, cols):
    df = pd.read_csv(DATA / name)
    df = df[(df.season >= 2010) & (df.lg == "NBA")]
    return df[KEYS + list(cols)].rename(columns=cols)


def player_team_seasons():
    """Every player, team, season stint (traded players have one row per team plus a total row)."""
    p = _load("Per_100_Poss.csv", PER100)
    s = _load("Player_Shooting.csv", SHOOT)
    b = _load("Player_Play_By_Play.csv", PBP)
    df = p.merge(s, on=KEYS, how="left").merge(b, on=KEYS, how="left")
    # Players with no attempts from a range have NaN there. Zero is the right value.
    fill_zero = ["ast3_rate", "corner3_share", "ast2_rate", "dunk_share"]
    df[fill_zero] = df[fill_zero].fillna(0)
    return df


def player_seasons(df=None):
    """One row per player season. Traded players keep their season total (2TM, 3TM...) row."""
    df = player_team_seasons() if df is None else df
    multi = df.team.str.match(r"\dTM")
    totals = df[multi]
    singles = df[~multi & ~df.set_index(["season", "player_id"]).index.isin(
        totals.set_index(["season", "player_id"]).index)]
    out = pd.concat([totals, singles], ignore_index=True)
    return out.dropna(subset=FEATURES).reset_index(drop=True)


if __name__ == "__main__":
    ps = player_seasons()
    print(ps.shape, ps.season.min(), ps.season.max())
    print(ps[FEATURES].describe().T[["mean", "std", "min", "max"]].round(2))
