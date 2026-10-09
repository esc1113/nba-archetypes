"""Make the README charts from the files analysis.py writes to output/."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster import OUT

FIG = OUT.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})

INK, MUTED, RISE, FALL = "#1f2328", "#c9ccd1", "#1f6feb", "#cf4b2b"


def evolution():
    evo = pd.read_csv(OUT / "archetype_minute_share_by_season.csv", index_col=0) * 100
    change = evo.iloc[-1] - evo.iloc[0]
    order = change.sort_values().index
    fig, axes = plt.subplots(3, 4, figsize=(11, 7), sharex=True, sharey=True)
    plt.setp(axes, ylim=(0, 24))
    for ax, a in zip(axes.flat, order):
        c = RISE if change[a] > 3 else FALL if change[a] < -3 else "#6e7781"
        ax.fill_between(evo.index, evo[a], color=c, alpha=0.12, lw=0)
        ax.plot(evo.index, evo[a], color=c, lw=2)
        ax.set_title(a, fontsize=10, loc="left")
        ax.text(0.98, 0.86, f"{evo[a].iloc[0]:.1f}% to {evo[a].iloc[-1]:.1f}%", ha="right",
                transform=ax.transAxes, fontsize=8.5, color=c)
        ax.set_xticks([2010, 2018, 2026], ["2010", "2018", "2026"])
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of minutes")
    fig.suptitle("Midrange and post up roles nearly vanished. Shooting forwards and wings took their minutes.",
                 fontweight="bold", fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.925, "Each archetype's share of all regular season minutes, 2009-10 to 2025-26 seasons",
             fontsize=9, color="#57606a")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "archetype_evolution.png", dpi=180)


def team_model():
    r = json.loads((OUT / "results.json").read_text())["team"]
    names = ["Archetype mix only", "Prior season talent only", "Talent + archetype mix"]
    vals = [r["r2_archetypes_only"], r["r2_talent_only"], r["r2_talent_plus_archetypes"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.barh(names, vals, color=[MUTED, RISE, RISE], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center")
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Out of sample R² for team point differential (leave one season out)")
    ax.set_title("Talent explains team results, archetype mix adds nothing")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "team_model.png", dpi=180)


def profiles():
    p = pd.read_csv(OUT / "archetype_profiles.csv", index_col=0)
    cols = {"pts100": "Points", "ast100": "Assists", "fg3a100": "3PA", "fta100": "FTA",
            "orb100": "Off reb", "drb100": "Def reb", "blk100": "Blocks", "stl100": "Steals",
            "rim_share": "Rim shots", "mid_share": "Midrange", "three_share": "Threes",
            "ast2_rate": "Assisted 2s", "corner3_share": "Corner 3s", "pos_index": "Position"}
    z = (p[list(cols)] - p[list(cols)].mean()) / p[list(cols)].std()
    z = z.loc[p.sort_values("pos_index").index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(cols)), list(cols.values()), rotation=40, ha="right")
    ax.set_yticks(range(len(z)), [f"{i}  (n={int(p.player_seasons[i])})" for i in z.index])
    ax.set_title("What each archetype does, relative to the other archetypes")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Standard deviations from archetype average")
    fig.tight_layout()
    fig.savefig(FIG / "archetype_profiles.png", dpi=180)


if __name__ == "__main__":
    evolution()
    team_model()
    profiles()
    print("saved to", FIG)
