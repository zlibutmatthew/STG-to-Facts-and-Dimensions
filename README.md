# STG-to-Facts-and-Dimensions

# Fact and Dimension Prediction

## Features:
1. Number of unique values
2. Number of NULL values
3. If ID, SID, AMT, SUM, AVG, or CALC is in the column name
4. Max, Min, and Average of the length of values in the columns
5. Data Type that is in the columns

### User input: **Staging Tables**

#### Description
Web_main.py runs a program to open a local host website, and it loads the saved model called 'created_model.pkl'. Once open, you can upload a staging table to the website, and it will tell you which columns are Facts and which are Dimensions. The model was created by adding the same features to the training data and training various models to test which performed the best. I determined this with GridSearch Cross Validation. The tested models include: Logistic Regression, Support Vector Machine, Decision Tree, and Random Forest Classification. It was found that a Random Forest Classification performed the best, and the file 'created_model.pkl' was created. I cannot upload the data or file I used to create the model because I'm afraid it would violate privacy. 
