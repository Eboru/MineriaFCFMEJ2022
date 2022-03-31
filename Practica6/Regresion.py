import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from scipy.stats import norm
import math
from tabulate import tabulate

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#Anderson-Darling
#Valores criticos   0.501, 0.57 , 0.684, 0.798, 0.95
#Significancia      15%  , 10%  , 5%   , 2.5% , 1%
def testNormality(name : str, sr: pd.Series):
    if sr.size==0:
        return
    sr = sr.sort_values()
    mean = sr.mean()
    std = sr.std()
    contador = 1
    cumValues = 0
    for _, val in sr.iteritems():
        yi = (val - mean) / std
        cumValues = cumValues + (2*contador-1)*(np.log(norm.cdf(yi))) + (2 * (sr.size-contador)+1)*np.log(1-norm.cdf(yi))
        contador +=1
    estadistico = -sr.size - (1/sr.size)*cumValues
    sstat,_,_ = ss.anderson(sr.array)
    #H0: Los datos son normales
    #H1: Los datos no son normales
    #Significancia del 1% -> valor critico 0.685
    #The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population that follows a particular distribution (normal).
    #If the returned statistic is larger than these critical values then for the corresponding significance level,
    #the null hypothesis that the data come from the chosen distribution can be rejected.
    if(estadistico>0.95): #Rechazamos h0
        print("Los datos no son normales para ", name, "[", estadistico,sstat, "]")
    else: #Aceptamos h0
        print("Los datos son normales para ", name, "[", estadistico,sstat, "]")


def spearmanRankCorrelation(name : str, sr1: pd.Series, sr2: pd.Series):
    if sr1.size == 0 or sr2.size==0:
        return
    if sr1.size != sr2.size:
        return
    sr1r = sr1.rank()
    sr2r = sr2.rank()
    di = sr1r-sr2r
    di2 = di*di
    sum = di2.sum()
    spearman = 1 - (6*sum)/(sr1.size*sr1.size*sr1.size - sr1.size)
    print("Spearman rank correlation coefficient:",spearman)

def pearsonCorrelation(name : str, sr1: pd.Series, sr2: pd.Series):
    if sr1.size == 0 or sr2.size==0:
        return
    if sr1.size != sr2.size:
        return
    cov = sr1.cov(sr2)
    varsr1 = sr1.var()
    varsr2 = sr2.var()
    pearson = cov/math.sqrt(varsr1*varsr2)
    print("Pearson correlation coefficient:", pearson)

def sign(x, reference):
    if x>reference:
        return 1
    if x<reference:
        return -1
    return 0





df = pd.read_csv("cleanDataPlayerInfo.csv")
heroes = pd.read_csv("heroes.csv")
match = pd.read_csv("cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero").join(match.set_index("match_id"), on="idPartida")



#Test against Spearman's rank correlation coefficient
spearmanRankCorrelation("Spearman's Correlation between xpPerMin and Kills", joinChar["kills"], joinChar["xpPerMin"])
#Test agains Pearson's correlation coefficient
pearsonCorrelation("Pearson's Correlation between xpPerMin and Kills", joinChar["kills"], joinChar["xpPerMin"])
#spearman -> posible relación monotona positiva
#pearson  -> posible relacion lineal positiva

modelo = sm.OLS(endog=joinChar["xpPerMin"], exog=sm.add_constant(joinChar["kills"], prepend=True)).fit()
print(modelo.summary())
prediccion = modelo.get_prediction(exog = sm.add_constant(joinChar["kills"], prepend=True)).summary_frame(alpha=0.05)

#Test F
#H0: yi = b0+ei
#h1: yi = b0+ b1xi + ei

#Como P del estadístido F es 0 hay suficiente evidencia para rechazar h0 por lo tanto hay una asociacion lineal siginficativa
#Información de prueba F
#https://online.stat.psu.edu/stat501/lesson/6/6.2

#Prueba Omnibus
#H0: b1 = 0
#H1: bi != 0
#Como P del estadístico omnibus es 0 hay suficiente evidencia para rechazar h0 por lo el coeficiente b1 debe ser distinto de 0
#Información de prueba omnibus
#https://www.statology.org/omnibus-test/

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot()
ax1.scatter(joinChar["kills"], joinChar["xpPerMin"])
ax1.plot(joinChar["kills"], prediccion["mean"], linestyle='-', label="Minimos Cuadrados", color="red")
plt.savefig("img/kills-xpPerMin.png")
plt.close()




print("\n")
#Test against Spearman's rank correlation coefficient
spearmanRankCorrelation("Spearman's Correlation between goldPerMin and xpPerMin", joinChar["goldPerMin"], joinChar["xpPerMin"])
#Test agains Pearson's correlation coefficient
pearsonCorrelation("Pearson's Correlation between goldPerMin and xpPerMin", joinChar["goldPerMin"], joinChar["xpPerMin"])
#spearman -> posible relación monotona positiva
#pearson  -> posible relacion lineal positiva

modelo = sm.OLS(endog=joinChar["xpPerMin"], exog=sm.add_constant(joinChar["goldPerMin"], prepend=True)).fit()
print(modelo.summary())
prediccion = modelo.get_prediction(exog = sm.add_constant(joinChar["goldPerMin"], prepend=True)).summary_frame(alpha=0.05)

#Test F
#H0: yi = b0+ei
#h1: yi = b0+ b1xi + ei

#Como P del estadístico F es 0 hay suficiente evidencia para rechazar h0 por lo tanto hay una asociacion lineal siginficativa
#Información de prueba F
#https://online.stat.psu.edu/stat501/lesson/6/6.2

#Prueba Omnibus
#H0: b1 = 0
#H1: bi != 0
#Como P del estadístico omnibus es  0 hay suficiente evidencia para rechazar h0 por lo el coeficiente b1 debe ser distinto de 0
#Información de prueba omnibus
#https://www.statology.org/omnibus-test/

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot()
ax1.scatter(joinChar["goldPerMin"], joinChar["xpPerMin"])
ax1.plot(joinChar["goldPerMin"], prediccion["mean"], linestyle='-', label="Minimos Cuadrados", color="red")
plt.ylabel("xpPerMin")
plt.xlabel("goldPerMin")
plt.savefig("img/goldPerMin-xpPerMin.png")
plt.close()

