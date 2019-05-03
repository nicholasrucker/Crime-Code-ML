import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from collections import defaultdict

dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

data = dataSet.iloc[:, [9, 10, 12, 13, 14, 20, 21, 25, 31]]
target = dataSet.iloc[:, 5].values

i = 0

for value in target:
	value = str(value)
	target[i] = re.sub('[^0-9]','', value)
	i = i + 1

dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)

scoreMap = defaultdict(list) 

for j in range (0, 150):
	for i in range (1, 26):		
		knn = KNeighborsClassifier(n_neighbors = i)
		knn.fit(dataTrain, targetTrain)
		predictions = knn.predict(dataTest)

		aS = accuracy_score(targetTest, predictions)

		scoreMap[i].append(aS)

		i = i + 1