from sklearn import svm
import pandas as pd
from datetime import datetime
import csv
import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier




def process_and_split_data(temp_df):
    split_percent_ratio=(100*12/24) # 50%training 
    traindata=temp_df.iloc[:int((len(temp_df)*split_percent_ratio)/(100))]
    testdata=temp_df.iloc[int((len(temp_df)*split_percent_ratio)/100)+1:]
    return traindata,testdata
    
temp_df = pd.read_csv("shampoo_sales.csv", index_col=False)
# splitting the data
traindata, testdata = process_and_split_data(temp_df)
# training data  
X_train=traindata.iloc[:,:-1].values
y_train=traindata.iloc[:,-1].values.tolist()
# testing data
X_test=(testdata.iloc[:,:-1]).values
y_test=(testdata.iloc[:,-1]).values.tolist()


#X_test = pd.DataFrame([(1,0,0,0,0)])
# prediction part 
dtree_model = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
print (X_test.shape)
print (dtree_predictions)

decision_tree_pkl_filename = 'shampoo_choosing.pkl'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(dtree_model, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()

