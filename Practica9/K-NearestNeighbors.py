import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import statistics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tabulate import tabulate

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

def euclidean_distance(p_1x: float, p_1y: float, p_2x: float, p_2y: float) -> float:
    return math.sqrt((((p_2x-p_1x) ** 2) + ((p_2y-p_1y)**2)))

#http://exponentis.es/ejemplo-de-clustering-con-k-means-en-python
def kmeansClusteringAndNearestNeighbors(df: pd.DataFrame, x: str, y: str, kClustering: int, kNeighbors: int, numeroDatosNuevos: int, mux : float, sigmax : float, muy : float, sigmay : float):

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
    kmeans = KMeans(n_clusters=kClustering).fit(df)
    centroids = kmeans.cluster_centers_

    #Etiquetamos cada row del data frame
    labels = kmeans.predict(df)
    df['label'] = labels



    #KNN
    datosPorAgrupar = pd.DataFrame({x: np.random.normal(mux, sigmax, numeroDatosNuevos), y: np.random.normal(muy, sigmay, numeroDatosNuevos)})
    clasificaciones = []
    for idx, row in datosPorAgrupar.iterrows():
        distancias = []
        for idx2, classifiedRow in df.iterrows():
            distancias.append({'indice' : idx2, 'distancia' : euclidean_distance(row[0], row[1], classifiedRow[0], classifiedRow[1])})
        distancias.sort(key=lambda x: x["distancia"])
        distancias = distancias[:kNeighbors]
        etiquetas = []
        for e in distancias:
            etiquetas.append(df["label"][e["indice"]])
        clasificaciones.append(statistics.mode(etiquetas))




    colores = ['r', 'g', 'b', 'y', 'c', 'm']

    #Asignamos un color dependiendo del indice de la etiqueta de labels
    asignar = []
    for row in labels:
        asignar.append(colores[row])

    #Asignamos colores a las clasificaciones de datos nuevos
    asignarDatosKNN = []
    for e in clasificaciones:
        asignarDatosKNN.append(colores[e])


    plt.scatter(df[x], df[y], c=asignar, s=1)
    plt.scatter(datosPorAgrupar[x], datosPorAgrupar[y], c="black", marker="^", s=50)
    plt.scatter(datosPorAgrupar[x], datosPorAgrupar[y], c=asignarDatosKNN, marker="^", s=20)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=20)
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
kmeansClusteringAndNearestNeighbors(dfDurationFirstBlood, "Duration", "First Blood", 4, 3, 5, 2500, 500, 150, 20)

dfDurationDireScore = pd.DataFrame({ 'Duration': match["duration"], 'Dire Score': match["dire_score"] })
kmeansClusteringAndNearestNeighbors(dfDurationDireScore, "Duration", "Dire Score", 4, 3, 5, 2500, 500, 40, 5)

dfDireScoreRadiantScore = pd.DataFrame({ 'Radiant Score': match["radiant_score"], 'Dire Score': match["dire_score"] })
kmeansClusteringAndNearestNeighbors(dfDireScoreRadiantScore, "Radiant Score", "Dire Score", 4, 3, 5, 40, 5, 40, 5)


