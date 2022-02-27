import numpy as np
import pandas as pd
import json
from tabulate import tabulate

ANCIENT_BOTTOM = np.int16(0b0000010000000000)
ANCIENT_TOP = np.int16(0b0000001000000000)
BOT_TIER_3 = np.int16(0b0000000100000000)
BOT_TIER_2 = np.int16(0b0000000010000000)
BOT_TIER_1 = np.int16(0b0000000001000000)
MID_TIER_3 = np.int16(0b0000000000100000)
MID_TIER_2 = np.int16(0b0000000000010000)
MID_TIER_1 = np.int16(0b0000000000001000)
TOP_TIER_3 = np.int16(0b0000000000000100)
TOP_TIER_2 = np.int16(0b0000000000000010)
TOP_TIER_1 = np.int16(0b0000000000000001)

BOT_RANGED = np.int8(0b00100000)
BOT_MELEE = np.int8(0b00010000)
MID_RANGED = np.int8(0b00001000)
MID_MELEE = np.int8(0b00000100)
TOP_RANGED = np.int8(0b00000010)
TOP_MELEE = np.int8(0b10000000)

DIRE = np.int8(0b10000000)


def check_tower_status(value: np.int16, tower_status: np.int16) -> bool:
    return value & tower_status


def check_barrack_status(value: np.int8, barrack_status: np.int8) -> bool:
    return value & barrack_status


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))


def clear_win(str: str) -> str:
    if str:
        return 1
    else:
        return 0


def clear_player_slot(value: np.int8) -> bool:
    return not (value & DIRE)


def make_hero_match_csv(my_df: pd.DataFrame) -> pd.DataFrame:
    return_df = pd.DataFrame(columns=["idHero", "idPartida", "isRadiant", "kills", "deaths", "assists", "lastHits", "denies",
                                "goldPerMin", "xpPerMin", "level", "nethWorth", "heroDamage", "towerDamage",
                                "heroHealing",
                                "goldSpent"])
    counter = 0
    for index, row in my_df.iterrows():
        rawJSON = row["players"].replace("'", "\"")
        playersData = json.loads(rawJSON)
        for player in playersData:
            playerInfo = pd.DataFrame([[player["hero_id"], row["match_id"], clear_player_slot(player["player_slot"]), player["kills"], player["deaths"],
                            player["assists"], player["last_hits"], player["denies"], player["gold_per_min"], player["xp_per_min"], player["level"],
                            player["net_worth"], player["hero_damage"], player["tower_damage"], player["hero_healing"], player["gold_spent"]]],
                            columns=["idHero", "idPartida", "isRadiant", "kills", "deaths", "assists",
                            "lastHits", "denies", "goldPerMin", "xpPerMin", "level", "nethWorth",
                            "heroDamage", "towerDamage", "heroHealing", "goldSpent"], index=[counter])
            return_df = pd.concat([return_df, playerInfo])
            counter = counter+1
    return return_df


# Dropna porque hay unas lineas literalmente vacias no se por que
df = pd.read_csv("dataRealBuenaUsaEsta.csv").dropna().drop(
    columns=["picks_bans", "pre_game_duration", "match_seq_num", "leagueid", "positive_votes",
             "negative_votes", "flags", "engine"])

df["radiant_win"] = df["radiant_win"].transform(clear_win)
pdf = make_hero_match_csv(df)
df = df.drop(columns=["players"])


df.to_csv("Practica2/cleanData.csv")
pdf.to_csv("Practica2/cleanDataPlayerInfo.csv")
print_tabulate(df.iloc[:10])
