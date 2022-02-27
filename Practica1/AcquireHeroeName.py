import urllib.request
import json
import csv
import time


request = urllib.request.Request(
    "https://api.opendota.com/api/heroes",
    data=None,
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    }
)
headStr = urllib.request.urlopen(request).read()
jsonInfo = json.loads(headStr)

file = open("heroes.csv", 'w', newline='')
csv_writer = csv.writer(file)

csv_writer.writerow(["id", "name", "primary_attr", "attack_type", "legs"])

for heroe in jsonInfo:
    csv_writer.writerow([heroe["id"], heroe["localized_name"], heroe["primary_attr"], heroe["attack_type"], heroe["legs"]])
file.close()
