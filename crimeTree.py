import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Reading the dataset 
dataSet = pd.read_csv("cleanData/cleanedCrimes.csv")

# Seperating data and target values 
data = dataSet.iloc[:, [9, 10, 12, 13, 14, 20, 21, 25, 31]]
target = dataSet.iloc[:, 5].values

# Splitting the data up for internal validation
dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, test_size = .2)
decisionTreeMachine = tree.DecisionTreeClassifier(criterion="gini")

# Fit and predict
decisionTreeMachine.fit(dataTrain, targetTrain)
predictions = decisionTreeMachine.predict(dataTest)

# Visualization of the accuracy of the random forest
print(accuracy_score(targetTest, predictions))