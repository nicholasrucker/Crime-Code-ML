import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics

dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

data = dataSet.iloc[:, [9, 10, 11, 12, 13, 14, 20, 21, 25, 27, 29, 30]]
target = dataSet.iloc[:, 5].values

data = data.values

kfold_machine = KFold(n_splits = 4)
kfold_machine.get_n_splits(data)
print(kfold_machine)

for i in range (1, 10):
	print("With", i, "clusters: ")
	for trainingIndex, testIndex in kfold_machine.split(data):
		print("Training: ", trainingIndex)
		print("Test: ", testIndex)
		dataTrain, dataTest = data[trainingIndex], data[testIndex]
		targetTrain, targestTest = target[trainingIndex], target[testIndex]
		
		knn = KNeighborsClassifier(n_neighbors = i)
		knn.fit(dataTrain, targetTrain)
		predictions = knn.predict(dataTest)
		
		print(metrics.r2_score(targestTest,predictions))

	i = i + 1