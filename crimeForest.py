import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

data = dataSet.iloc[:, [9, 10, 11, 12, 13, 14, 20, 21, 25, 27, 29, 30, 31]]
target = dataSet.iloc[:, 5].values

dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)

for i in range (1,50):
	print("Now using", i, "estimators")
	RandomForestMachine = RandomForestClassifier(n_estimators = i)
	RandomForestMachine.fit(dataTrain, targetTrain)

	predictions = RandomForestMachine.predict(dataTest)

	print(accuracy_score(targetTest, predictions))