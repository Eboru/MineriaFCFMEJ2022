import pandas as pd
import numpy as np
from tabulate import tabulate


def clasify_time(usage: int) -> str:
    if usage <= 900:
        return "VERY_LOW"
    if 900 < usage <= 1800:
        return "LOW"
    if 1800 < usage <= 2700:
        return "MEDIUM"
    if 2700 < usage <= 3600:
        return "HIGH"
    if usage > 3600:
        return "VERY_HIGH"


df = pd.read_csv("Practica3\cleanData.csv")
newColumn = []

for index, row in df.iterrows():
    newColumn.append(clasify_time(row["duration"]))

df["timeClassification"] = newColumn

print(df)
df.to_csv("Practica3/cleanData.csv")