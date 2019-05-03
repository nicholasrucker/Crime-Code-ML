import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import re

# Reading the dataset 
dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

# Seperating data and target values 
data = dataSet.iloc[:, [9, 10, 12, 13, 14, 20, 21, 25, 31]]
target = dataSet.iloc[:, 5].values

# Now I am going to use a simple regex to remove the letters from the IUCR
# (This needs to be done for KNN and Linear Regression)
i = 0

for value in target:
	value = str(value)
	target[i] = re.sub('[^0-9]','', value)
	i = i + 1

# Splitting the data for internal validation
dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)

# Fit, predicit, and a visualization of accuracy
linear_machine = linear_model.LinearRegression()
linear_machine.fit(dataTrain, targetTrain)
prediction = linear_machine.predict(dataTest)


print(metrics.r2_score(targetTest, prediction))