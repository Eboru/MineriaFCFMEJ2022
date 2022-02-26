import urllib.request
import json
import csv
import time

apiKey = "2C31978175FAEE9AF2857E3F8FECAB9F"

headStr = urllib.request.urlopen(
    "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V001/?key=2C31978175FAEE9AF2857E3F8FECAB9F"
    "&min_players=10&game_mode=4&matches_requested=1").read()
headStore = json.loads(headStr)
headSeqNum = headStore["result"]["matches"][0]["match_seq_num"]
totalMatches = 10000
matchesPerRequest = 500
passedMatchesCounter = 0
requestsMade = 0

file = open("data2.csv", 'w')
csv_writer = csv.writer(file)
headerWrote = 0

averageRequestSpeed = 0

while passedMatchesCounter < totalMatches:
    seqNum = headSeqNum - matchesPerRequest * (requestsMade + 1)
    url = "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/V001?key" \
          "={}&start_at_match_seq_num={}&matches_requested={}".format(apiKey, seqNum, matchesPerRequest)
    jsonStr = urllib.request.urlopen(url).read()
    requestStorage = json.loads(jsonStr)
    matches = requestStorage["result"]["matches"]

    inRequestPassed = 0
    for i in range(0, len(matches)):
        if not (matches[i]["game_mode"] == 1 or matches[i]["game_mode"] == 22):
            # print("Skipped one match, bad game mode")
            continue
        if headerWrote == 0:
            csv_writer.writerow(matches[i].keys())
            headerWrote = 1
        csv_writer.writerow(matches[i].values())
        inRequestPassed += 1

    requestsMade += 1
    passedMatchesCounter += inRequestPassed

    if averageRequestSpeed == 0:
        averageRequestSpeed = inRequestPassed
    else:
        averageRequestSpeed = (averageRequestSpeed + inRequestPassed) / 2

    print("Average request speed :" + str(averageRequestSpeed * 2) + " rpm")
    eta = (totalMatches - passedMatchesCounter) / (averageRequestSpeed * 2)
    print("ETA : " + str(eta) + " minutes")
    if passedMatchesCounter < totalMatches:
        time.sleep(30)
file.close()
print("Request made {}".format(requestsMade))
