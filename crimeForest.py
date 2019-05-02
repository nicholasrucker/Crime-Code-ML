import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

data = dataSet.iloc[:, [9, 10, 12, 13, 14, 20, 21, 25, 31]]
target = dataSet.iloc[:, 5].values

dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)
randomForestMachine = RandomForestClassifier(n_estimators = 21)
randomForestMachine.fit(dataTrain, targetTrain)

predictions = randomForestMachine.predict(dataTest)

print(accuracy_score(targetTest, predictions))