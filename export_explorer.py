"""Write docs/index.html: a self contained player similarity explorer (GitHub Pages ready)."""
import json

import numpy as np
import pandas as pd

from analysis import PROTOTYPES, fit_archetypes, profiles, similarity_index, evolution
from cluster import OUT

ROOT = OUT.parent
BLURB = {
    "Primary Creator": "High usage stars who score, draw fouls, and create for others.",
    "Floor General": "Pass first guards who run the offense and pressure the ball.",
    "Secondary Scorer": "Perimeter scorers who shoot threes and attack closeouts.",
    "Spot Up Shooter": "Catch and shoot specialists. Most threes are assisted.",
    "Two Way Wing": "Corner threes, steals, and cutting. Low usage, lots of defense.",
    "Stretch Forward": "Forwards who rebound and space the floor from three.",
    "Midrange Scorer": "Wings and forwards who live between the paint and the arc.",
    "Slashing Forward": "Athletic forwards who score at the rim and crash the glass.",
    "Hub Big": "Bigs who score, rebound, and run the offense through the post or elbow.",
    "Post Scoring Big": "Back to the basket bigs with no three point shot.",
    "Rim Running Big": "Lob finishers and shot blockers. Most shots are dunks.",
}


def main():
    train, pool, Xp, *_ = fit_archetypes()
    V = similarity_index(pool, Xp)
    arche = list(PROTOTYPES)
    idx = {a: i for i, a in enumerate(arche)}
    evo = evolution(pool)[arche]
    prof = profiles(train)
    examples = {}
    for a in arche:
        t = pool[(pool.archetype == a) & (pool.mp >= 1500)].sort_values("mp", ascending=False)
        t = t.drop_duplicates("player_id").head(4)
        examples[a] = [f"{r.player} ({r.season})" for r in t.itertuples()]

    rows = []
    for i, r in pool.iterrows():
        rows.append([r.player, int(r.season), r.team, int(r.mp), idx[r.archetype],
                     round(r.pts100, 1), round(r.ast100, 1), round(r.fg3a100, 1),
                     round(r.drb100 + r.orb100, 1), round(r.rim_share * 100), round(r.three_share * 100),
                     [round(float(x), 3) for x in V[i]]])
    data = {
        "archetypes": [{"name": a, "blurb": BLURB[a], "examples": examples[a],
                        "share": [round(float(x) * 100, 2) for x in evo[a]],
                        "n": int(prof.loc[a, "player_seasons"])} for a in arche],
        "seasons": [int(s) for s in evo.index],
        "rows": rows,
    }
    tpl = (ROOT / "src" / "explorer_template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html)
    print("rows", len(rows), "bytes", len(html))


if __name__ == "__main__":
    main()
