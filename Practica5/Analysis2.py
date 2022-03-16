import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

def signTest(name : str, sr1: pd.Series):
    if sr1.size == 0:
        return
    mediaMuestral = sr1.mean()
    sr1Clon = sr1.transform(lambda x: sign(x, mediaMuestral))
    positivos = sr1Clon.where(sr1Clon>0).count()
    a = 0.05
    normInv = norm.ppf(a)
    z = (positivos - sr1.size * 0.5) / (math.sqrt(sr1.size * 0.5 * (1 - 0.5)))
    #Probamos
    # H0 u<=mediaMuestarl
    # H1 u>mediaMuestral
    #Rechazamos H0 si z > Za
    if z>normInv:
        print(name, ": Rechazamos la hipotesis nula -> u >", mediaMuestral)
    else:
        print(name, ": Aceptamos la hipotesis nula -> u <=", mediaMuestral)

def wilcoxowSignTest(name : str, sr1: pd.Series, sr2: pd.Series):
    if sr1.size == 0 or sr2.size==0:
        return
    if sr1.size != sr2.size:
        return



df = pd.read_csv("cleanDataPlayerInfo.csv")
heroes = pd.read_csv("heroes.csv")
match = pd.read_csv("cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero")


#Normality test for deaths for each hero
#for index_h, row_h in heroes.iterrows():
#    mapa = {"k": [], "d": [], "a": []}
#    queryStr = "name == \"" + row_h["name"]+"\""
#    df_query = joinChar.query(queryStr)
#    testNormality(row_h["name"], df_query["deaths"])

#Plot xpPerMin distribution
#fig = plt.figure(figsize=(16, 9))
#ax1 = fig.add_subplot()
#ax1.hist(joinChar["xpPerMin"], label="xpPerMin", alpha=1, bins=25)
#ax1.set_title("xpPerMin")
#plt.savefig("img/xpPerMin.png")
#plt.close()
# Normality test for xp per min (Anderson-Darling)
#testNormality("xpPerMin", joinChar["xpPerMin"])

#Plot kills distribution
#fig = plt.figure(figsize=(16, 9))
#ax1 = fig.add_subplot()
#ax1.hist(joinChar["kills"], label="kills", alpha=1, bins=25)
#ax1.set_title("Kills")
#plt.savefig("img/kills.png")
#plt.close()
## Normality test for kills (Anderson-Darling)
#testNormality("kills", joinChar["kills"])

#Plot kills-xpPerMin histogram
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot()
ax1.hist2d(joinChar["kills"], joinChar["xpPerMin"], label="kills", alpha=1, bins=50)
ax1.set_title("Kills")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.savefig("img/kills-xpPerMin.png")
plt.close()
#Test against Spearman's rank correlation coefficient
spearmanRankCorrelation("Spearman's Correlation between xpPerMin and Kills", joinChar["kills"], joinChar["xpPerMin"])
#Test agains Pearson's correlation coefficient
pearsonCorrelation("Pearson's Correlation between xpPerMin and Kills", joinChar["kills"], joinChar["xpPerMin"])
#spearman -> posible relación monotona positiva
#pearson  -> posible relacion lineal positiva
signTest("Kills", joinChar["kills"])
signTest("XpPerMin", joinChar["xpPerMin"])