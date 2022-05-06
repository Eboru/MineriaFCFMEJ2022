import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tabulate import tabulate

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#http://exponentis.es/ejemplo-de-clustering-con-k-means-en-python
def kmeansClustering(df: pd.DataFrame, x: str, y: str, k: int):

    #Como decidir # de centroides
    #Usamos grafica elbow y revisamos donde esta el codo
    nc = range(1, 30)
    kmeans = [KMeans(n_clusters=i) for i in nc]
    score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Suma de los errores cuadráticos')
    plt.plot(nc, score)
    plt.savefig("img/score_"+x+"_"+y+".png")
    plt.close()

    # Centroides
    kmeans = KMeans(n_clusters=k).fit(df)
    centroids = kmeans.cluster_centers_

    #Etiquetamos cada row del data frame
    labels = kmeans.predict(df)
    df['label'] = labels

    #


    colores = ['r', 'g', 'b', 'y', 'c', 'm']

    #Asignamos un color dependiendo del indice de la etiqueta de labels
    asignar = []
    for row in labels:
        asignar.append(colores[row])
    plt.scatter(df[x], df[y], c=asignar, s=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=20)  # Marco centroides.
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('K-means clustering')
    plt.savefig("img/clustering_"+x+"_"+y+".png")
    plt.close()


df = pd.read_csv("cleanDataPlayerInfo.csv")
heroes = pd.read_csv("heroes.csv")
match = pd.read_csv("cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero").join(match.set_index("match_id"), on="idPartida")


dfDurationFirstBlood = pd.DataFrame({ 'Duration': match["duration"], 'First Blood': match["first_blood_time"] })
kmeansClustering(dfDurationFirstBlood, "Duration", "First Blood", 4)

dfDurationDireScore = pd.DataFrame({ 'Duration': match["duration"], 'Dire Score': match["dire_score"] })
kmeansClustering(dfDurationDireScore, "Duration", "Dire Score", 4)

dfDireScoreRadiantScore = pd.DataFrame({ 'Radiant Score': match["radiant_score"], 'Dire Score': match["dire_score"] })
kmeansClustering(dfDireScoreRadiantScore, "Radiant Score", "Dire Score", 4)



