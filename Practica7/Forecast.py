import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import stats
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


def objectiveFunctionSqrt(x, a, b, c):
    return pd.Series(x).transform(lambda value: sqrtFunction(value,a,b,c))

def objectiveFunctionLinear(x, a, b):
    return pd.Series(x).transform(lambda value: linearFunction(value, a, b))

def linearFunction(x, a, b):
    return a*x + b

def sqrtFunction(x, a, b, c):
    return a*(x**(1/c)) + b

#https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb




df = pd.read_csv("cleanDataPlayerInfo.csv")
heroes = pd.read_csv("heroes.csv")
match = pd.read_csv("cleanData.csv")

# To avoid join errors
df.rename({"Unnamed: 0": "idPlayerInfo"}, axis="columns", inplace=True)
match.rename({"Unnamed: 0": "idMatchInfo"}, axis="columns", inplace=True)

joinMatch = df.join(match.set_index("match_id"), on="idPartida").join(heroes.set_index("id"), on="idHero")
joinChar = df.join(heroes.set_index("id"), on="idHero").join(match.set_index("match_id"), on="idPartida")




#Test against Spearman's rank correlation coefficient
spearmanRankCorrelation("Spearman's Correlation between goldPerMin and xpPerMin", joinChar["goldPerMin"], joinChar["xpPerMin"])
#Test agains Pearson's correlation coefficient
pearsonCorrelation("Pearson's Correlation between goldPerMin and xpPerMin", joinChar["goldPerMin"], joinChar["xpPerMin"])
#spearman -> posible relación monotona positiva
#pearson  -> posible relacion lineal positiva







#popt, pcov = curve_fit(objectiveFunctionSqrt, joinChar["goldPerMin"], joinChar["xpPerMin"], bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, 32)))
#rootXPoints = np.linspace(100, 1400, num=1300)
#predictedCurve = objectiveFunctionSqrt(rootXPoints, *popt)
##print("Curve is %.2fx^1/2+%.2f" % (popt[0], popt[1]))
#print(popt)
#
#residuals = joinChar["xpPerMin"] - joinChar["xpPerMin"].transform(lambda x: sqrtFunction(x, *popt))
#ss_residual = residuals.transform(lambda x: x**2).sum()
#mean = joinChar["xpPerMin"].mean()
#ss_total = joinChar["xpPerMin"].transform(lambda x: (x-mean)**2).sum()
#print("R^2 =",  1 - (ss_residual/ss_total))
#lpb, upb = predband(rootXPoints, joinChar["goldPerMin"], joinChar["xpPerMin"], popt, sqrtFunction, conf=0.99)







#fig = plt.figure(figsize=(16, 9))
#ax1 = fig.add_subplot()
#ax1.scatter(joinChar["goldPerMin"], joinChar["xpPerMin"])
#ax1.plot(rootXPoints, predictedCurve, linestyle='-', label="Minimos Cuadrados", color="red")
#ax1.plot(rootXPoints, lpb, linestyle='--', color="blue")
#ax1.plot(rootXPoints, upb, linestyle='--', color="blue")
#plt.ylabel("xpPerMin")
#plt.xlabel("goldPerMin")
#plt.savefig("img/goldPerMin-xpPerMin.png")
#plt.close()


popt, pcov = curve_fit(objectiveFunctionLinear, joinChar["goldSpent"], joinChar["nethWorth"], bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
rootXPoints = np.linspace(100, 70000, num=1300)
predictedCurve = objectiveFunctionLinear(rootXPoints, *popt)
#print("Curve is %.2fx^1/2+%.2f" % (popt[0], popt[1]))
print(popt)

residuals = joinChar["nethWorth"] - joinChar["nethWorth"].transform(lambda x: linearFunction(x, *popt))
ss_residual = residuals.transform(lambda x: x**2).sum()
mean = joinChar["nethWorth"].mean()
ss_total = joinChar["nethWorth"].transform(lambda x: (x-mean)**2).sum()
print("R^2 =",  1 - (ss_residual/ss_total))
lpb, upb = predband(rootXPoints, joinChar["goldSpent"], joinChar["nethWorth"], popt, linearFunction, conf=0.9999)

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot()
ax1.scatter(joinChar["goldSpent"], joinChar["nethWorth"])
ax1.plot(rootXPoints, predictedCurve, linestyle='-', label="Minimos Cuadrados", color="red")
ax1.plot(rootXPoints, lpb, linestyle='--', color="red")
ax1.plot(rootXPoints, upb, linestyle='--', color="red")
plt.ylabel("nethWorth")
plt.xlabel("goldSpent")
plt.savefig("img/goldSpent-netWorth.png")
plt.close()
