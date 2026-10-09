#!/usr/bin/env bash
# Download the public datasets this project uses into data/
set -euo pipefail
mkdir -p data && cd data
BBR=https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data
curl -fsSL "$BBR/Per%20100%20Poss.csv" -o Per_100_Poss.csv
curl -fsSL "$BBR/Player%20Shooting.csv" -o Player_Shooting.csv
curl -fsSL "$BBR/Player%20Play%20By%20Play.csv" -o Player_Play_By_Play.csv
curl -fsSL https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/regular_season_totals_2010_2024.csv -o regular_season_totals_2010_2024.csv
curl -fsSL https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_player.csv -o modern_RAPTOR_by_player.csv
echo "done"
