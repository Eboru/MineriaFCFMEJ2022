import pandas as pd
import numpy as np
from tabulate import tabulate


def match_average(usage: int) -> float:
    return float(usage / matchCount)


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))


df = pd.read_csv("Practica3\cleanDataPlayerInfo.csv")
heroes = pd.read_csv("Practica3\heroes.csv")
match = pd.read_csv("Practica3\cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero")

df_analisisChar = joinChar.groupby(["name"])[["kills", "deaths", "assists"]].mean().sort_values(by=["kills"], ascending=False)
df_analisis_tower_damage = joinChar.groupby(["name"])[["towerDamage", "nethWorth"]].mean().sort_values(by=["towerDamage"], ascending=False)

matchCount = joinMatch[["idMatchInfo"]].count().min()
df_popularity_per_match = joinChar.groupby(["name"])[["idPartida"]].count()
df_popularity_per_match["idPartida"] = df_popularity_per_match["idPartida"].transform(match_average)
df_popularity_per_match = df_popularity_per_match.sort_values(by=["idPartida"], ascending=False)
df_popularity_per_match.rename({"idPartida": "Porcentaje de eleccion"}, axis="columns", inplace=True)

df_matchDurationAndChar = joinMatch.groupby(["name", "timeClassification"])[["idPartida"]].count()
df_matchDurationAndChar.rename({"idPartida": "Cantidad"}, axis="columns", inplace=True)

print_tabulate(df_popularity_per_match)
print("\n")
print_tabulate(df_analisisChar)
print("\n")
print_tabulate(df_analisis_tower_damage)
print("\n")
print_tabulate(df_matchDurationAndChar)
