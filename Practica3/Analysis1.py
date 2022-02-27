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

df_charVictoria = joinMatch.query("isRadiant == True").groupby(["name"])[["radiant_win"]].sum() #contar ocurrencia de victoria
df_charTotal = joinMatch.groupby(["name"])[["radiant_win"]].count() #contar ocurrencias totales
df_analisisPorcentajeVictoria = df_charVictoria.div(df_charTotal).sort_values(by=["radiant_win"], ascending=False)
df_analisisPorcentajeVictoria.rename({"radiant_win": "Promedio de victorias al elegir el personaje"}, axis="columns", inplace=True)

#df_analisisPorcentajeVictoria = joinMatch.groupby(["name"])[["radiant_win"]].mean().sort_values(by=["radiant_win"], ascending=False)
#df_analisisPorcentajeVictoria.rename({"radiant_win": "Promedio de victorias radiant si esta el personaje"}, axis="columns", inplace=True)

matchCount = joinMatch[["idMatchInfo"]].count().min() #Contamos todas las partidas
df_popularity_per_match = joinChar.groupby(["name"])[["idPartida"]].count() #Agrupamos por nombre y contamos cuantas veces aparece
df_popularity_per_match["idPartida"] = df_popularity_per_match["idPartida"].transform(match_average) #Dividimos
df_popularity_per_match = df_popularity_per_match.sort_values(by=["idPartida"], ascending=False)
df_popularity_per_match.rename({"idPartida": "Porcentaje de eleccion global"}, axis="columns", inplace=True)

df_matchDurationAndChar = joinMatch
VERY_LOW = []
LOW = []
MEDIUM = []
HIGH = []
VERY_HIGH = []

for index, row in joinMatch.iterrows():

    if row["timeClassification"] == "VERY_LOW":
        VERY_LOW.append(1)
    else:
        VERY_LOW.append(0)

    if row["timeClassification"] == "LOW":
        LOW.append(1)
    else:
        LOW.append(0)

    if row["timeClassification"] == "MEDIUM":
        MEDIUM.append(1)
    else:
        MEDIUM.append(0)

    if row["timeClassification"] == "HIGH":
        HIGH.append(1)
    else:
        HIGH.append(0)

    if row["timeClassification"] == "VERY_HIGH":
        VERY_HIGH.append(1)
    else:
        VERY_HIGH.append(0)

df_matchDurationAndChar["VERY_LOW"] = VERY_LOW
df_matchDurationAndChar["LOW"] = LOW
df_matchDurationAndChar["MEDIUM"] = MEDIUM
df_matchDurationAndChar["HIGH"] = HIGH
df_matchDurationAndChar["VERY_HIGH"] = VERY_HIGH

df_matchDurationAndChar = joinMatch.groupby(["name"])[["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]].sum().sort_values(by=["HIGH"], ascending=False)
df_matchDurationAndChar.rename({"idPartida": "Cantidad"}, axis="columns", inplace=True)

df_charDesviation = joinChar.groupby(["name"])[["kills", "deaths", "assists"]].std().sort_values(by=["kills"], ascending=False)

print_tabulate(df_analisisPorcentajeVictoria)
print("\n")
print_tabulate(df_popularity_per_match)
print("\n")
print_tabulate(df_analisisChar)
print("\n")
print_tabulate(df_analisis_tower_damage)
print("\n")
print_tabulate(df_matchDurationAndChar)
print("\n")
print_tabulate(df_charDesviation)