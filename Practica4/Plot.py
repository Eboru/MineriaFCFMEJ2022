import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tabulate import tabulate


def match_average(usage: int) -> float:
    return float(usage / matchCount)


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

df = pd.read_csv("cleanDataPlayerInfo.csv")
heroes = pd.read_csv("heroes.csv")
match = pd.read_csv("cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero")


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

df_matchCountPerChar = joinMatch.groupby(["name"])[["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]].count()
df_matchDurationAndChar = joinMatch.groupby(["name"])[["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]].sum().div(df_matchCountPerChar)
df_matchDurationAndChar.rename({"idPartida": "Cantidad"}, axis="columns", inplace=True)

#Relacion denegados-kills
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot()
ax1.hist2d(x=joinMatch["xpPerMin"], y=joinMatch["goldPerMin"], bins=100, cmap=plt.cm.plasma)
x = np.linspace(ax1.get_xlim(),100)
y = x
ax1.plot(x, y)
plt.ylabel("Gold Per Min")
plt.xlabel("XP Per Min")
plt.savefig("img/hist2d.png")
plt.close()



#KDA con Desviación
analysis_kda = joinChar.groupby(["name"])[["kills", "deaths", "assists"]].agg([np.mean, np.std])
for index, row in analysis_kda.iterrows():
    row.unstack().plot(kind = "barh", y = "mean", legend = True, xerr = "std", title = index + " avg and std", color='purple')
    plt.savefig("img/avg_std/"+index+"_avg_std.png")
    plt.close()



#Scatter procentaje de clasificación de duración
df_matchDurationAndChar.reset_index(inplace=True)
df_matchDurationAverage = df_matchDurationAndChar[["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]].mean().to_frame(name="mean")
df_matchDurationAverage.reset_index(inplace=True)
fig = plt.figure(figsize=(32,18))
ax1 = fig.add_subplot()
ax1.scatter(x=df_matchDurationAndChar["name"], y=df_matchDurationAndChar["VERY_LOW"], label='Very Low', s=100)
ax1.scatter(x=df_matchDurationAndChar["name"], y=df_matchDurationAndChar["LOW"], label='Low', s=100)
ax1.scatter(x=df_matchDurationAndChar["name"], y=df_matchDurationAndChar["MEDIUM"], label='Medium', s=100)
ax1.scatter(x=df_matchDurationAndChar["name"], y=df_matchDurationAndChar["HIGH"], label='High', s=100)
ax1.scatter(x=df_matchDurationAndChar["name"], y=df_matchDurationAndChar["VERY_HIGH"], label='Very High', s=100)
ax1.plot(ax1.get_xlim(), [df_matchDurationAverage["mean"], df_matchDurationAverage["mean"]])
ax1.legend(loc='upper left')
ax1.xaxis.labelpad = 20
plt.ylabel("Procentaje de clasificación de duración")
plt.xlabel("Heroe")
plt.xticks(rotation=90)
plt.savefig("img/scatter.png")
plt.close()

#Promedio first blood
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot()
match2 = match
match2["first_blood_time"] = match2["first_blood_time"].transform(lambda x: x/60)
match2 = match2.groupby(["timeClassification"])[["first_blood_time", "timeClassification"]].mean()
match2.reset_index(inplace=True)
ax1.bar(match2["timeClassification"], match2["first_blood_time"])
plt.ylabel("Tiempo Promedio Primera Sangre (minutos)")
plt.xlabel("Duracion de la partida")
plt.savefig("img/duracion_partida_primera_sangre.png")
plt.close()


for index_h, row_h in heroes.iterrows():
    mapa = {"k": [], "d": [], "a": []}
    queryStr = "name == \"" + row_h["name"]+"\""
    df_query = joinChar.query(queryStr)
    mapa["k"].append(df_query["kills"])
    mapa["d"].append(df_query["deaths"])
    mapa["a"].append(df_query["assists"])

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot()
    ax1.hist(mapa["k"], label="Kills", alpha=1, bins=25)
    ax1.set_title("Kills")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.savefig("img/dist/k_"+row_h["name"] +"_distribution.png")
    plt.close()

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot()
    ax1.hist(mapa["d"], label="Deaths", alpha=1, bins=25)
    ax1.set_title("Deaths")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.savefig("img/dist/d_"+row_h["name"] +"_distribution.png")
    plt.close()

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot()
    ax1.hist(mapa["a"], label="Assists", alpha=1, bins=25)
    ax1.set_title("Assists")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.savefig("img/dist/a_"+row_h["name"] +"_distribution.png")
    plt.close()

