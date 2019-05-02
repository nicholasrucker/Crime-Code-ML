import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import re

dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

data = dataSet.iloc[:, [9, 10, 12, 13, 14, 20, 21, 25, 31]]
target = dataSet.iloc[:, 5].values

i = 0

for value in target:
	value = str(value)
	target[i] = re.sub('[^0-9]','', value)
	i = i + 1

dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)

linear_machine = linear_model.LinearRegression()
linear_machine.fit(dataTrain, targetTrain)
prediction = linear_machine.predict(dataTest)


print(metrics.r2_score(targetTest, prediction))