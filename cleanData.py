import pandas as pd
import os
import random

# The data has 30 columns and 6,854,115 observations to start with
dataFrame = pd.read_csv("Crimes_-_2001_to_present.csv")

# Now we are going to drop observations that have any missing values
dataFrame.dropna(inplace=True)

# Now that the dataset is loaded and missing observations were dropped,
#	I am just going to randomize the observations
dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

# We are only going to keep the first 10,000 observations.
# 10,000 should be enough to train a machine since everything is randomized
dataFrame = dataFrame.head(10000)

crimeSet = {'ARSON'}

for index, row in dataFrame.iloc[1:].iterrows():
	crimeSet.add(row['Primary Type'])

crimeDict = {value : round(random.random() * 100) for value in crimeSet}
df2 = pd.DataFrame()

# Now we just need to add the key-value pair to a data frame and combine it with the first.
for i in range (len(dataFrame.index)):
	df2 = df2.append({
								'crimeCode' : crimeDict[dataFrame.loc[i,'Primary Type']],
								}, ignore_index = True)

# Lets create a directory so the cleaned data stored seperatly
if not os.path.exists("cleanData"):
	os.mkdir("cleanData")

dataFrame.join(df2, how = 'outer').to_csv("cleanData/cleanedCrimes.csv")