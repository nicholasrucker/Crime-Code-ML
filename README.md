Before trying to run any of the code make sure you have the following packages:
>pandas, regex, and sklearn

Below are the python3 commands to install any of the packages you may not have
```
pip install pandas
pip install regex
pip install scikit-learn
```

The data-set used in this project is available here
- 
- https://wetransfer.com/downloads/b7fc20cb9d54ee20e0b2fae325225bd120190428122621/ac6553

I am trying to train a variety of machines to learn the unique crime codes of Chicago 
-

File breakdown
-

cleanData.py
-
- This file is what houses the code that cleans the Chicago crime dataset into a more workable size.
- Upon running this script:
  - A new directory for the clean data will be made
  - Entries with any missing value will be dropped
  - The dataset will be randomized to help ensure a random sample
  - The dataset will be reduced to 10,000 observations from 6,000,000
  - Next a new column will be added which is a numerical code for the crime
  - Lastly, the new dataset will be written to a file called 'cleanedCrimes.csv' which will be housed in the 'cleanData' directory
- To execute the code use the following python3 command
```
python3 cleanData.py
```

**Make sure to run cleanData.py before running any of the machine learning programs**

crimeForest.py
-
- This file is where a random forest machine is used to predict the crime IUCR
- Internal validation is used to test the accuracy and the accuracy score will be printed in the console.
- More about the machine can be found in the writre up
- To execute the code use the following python3 command
```
python3 crimeForest.py
```

crimeKNN.py
-
- This file is where a KNN classifier is used to predict the crime IUCR
- Internal validation is used to test the accuracy and the accuracy score will be printed in the console.
- More about the machine can be found in the writre up
- To execute the code use the following python3 command
```
python3 crimeKNN.py
```

crimeLine.py
-
- This file is where a simple linear regression is used to predict the crime IUCR
- Internal validation is used to test the accuracy and the r<sup>2</sup> will be printed in the console.
- More about the machine can be found in the writre up
- To execute the code use the following python3 command
```
python3 crimeLine.py
```

crimeTree.py
-
- This file is where a decision tree is used to predict the crime IUCR
- Internal validation is used to test the accuracy and the accuracy score will be printed in the console.
- More about the machine can be found in the writre up
- To execute the code use the following python3 command
```
python3 crimeTree.py
```

