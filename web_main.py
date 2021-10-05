import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

st.write("""
# Fact and Dimension Prediction

## Features:
1. Number of unique values
2. Number of NULL values
3. If ID, SID, AMT, SUM, AVG, or CALC is in the column name
4. Max, Min, and Average of the length of values in the columns
5. Data Type that is in the columns

### User input: **Staging Tables**
""")
uploaded_file = st.file_uploader('Upload your Excel file here')
if uploaded_file is not None:
    result_df = pd.read_excel(uploaded_file)

    #adding the column name to a row
    ls=pd.Series(result_df.columns.values)
    #name of column
    name_dic = {}
    for x in ls:
        name_dic[x]= x
    result_df = result_df.append(name_dic, ignore_index=True)

    #adding the number of unique values
    ph = result_df.nunique()
    ls_unique = []
    ls_name = []
    for name in result_df.columns:
        ls_name.append(name)
    for item in ph:
        ls_unique.append(item)
    unique_dic = {ls_name[i]: ls_unique[i] for i in range(len(ls_name))}
    result_df = result_df.append(unique_dic, ignore_index=True)

    #adding amount of null values
    ph1 = result_df.isnull().sum()
    ls_null = []
    ls_name = []
    for name in result_df.columns:
        ls_name.append(name)
    for item in ph1:
        ls_null.append(item)
    null_dic = {ls_name[i]: ls_null[i] for i in range(len(ls_name))}
    result_df = result_df.append(null_dic, ignore_index=True)

    #checks if '_ID' is in the column name
    id_dic = {}
    for name in result_df.columns:
        if '_ID' in name:
            id_dic[name] = 1
        else:
            id_dic[name]=0
    result_df = result_df.append(id_dic, ignore_index=True)

    #checks if 'SID' is in the column name
    sid_dic = {}
    for name in result_df.columns:
        if 'SID' in name:
            sid_dic[name] = 1
        else:
            sid_dic[name]=0
    result_df = result_df.append(sid_dic, ignore_index=True)

    #checks if 'AMT' is in the column name
    amt_dic = {}
    for name in result_df.columns:
        if 'AMT' in name:
            amt_dic[name] = 1
        else:
            amt_dic[name]=0
    result_df = result_df.append(amt_dic, ignore_index=True)

    #checks if 'SUM' is in the name
    sum_dic = {}
    for name in result_df.columns:
        if 'SUM' in name:
            sum_dic[name] = 1
        else:
            sum_dic[name]=0
    result_df = result_df.append(sum_dic, ignore_index=True)

    #checks if 'AVG' is in the name
    avg_dic = {}
    for name in result_df.columns:
        if 'AVG' in name:
            avg_dic[name] = 1
        else:
            avg_dic[name]=0
    result_df = result_df.append(avg_dic, ignore_index=True)

    #checks if 'CALC' is in the name
    calc_dic = {}
    for name in result_df.columns:
        if 'CALC' in name:
            calc_dic[name] = 1
        else:
            calc_dic[name]=0
    result_df = result_df.append(calc_dic, ignore_index=True)

    #finds the max length value in the columns
    max_len_dic = dict( 
        [
            #create a tuple such that (column name, max length of values in column)
            (v, result_df[v].apply(lambda r: len(str(r))).max()) 
                for v in result_df.columns.values #iterates over all column values
        ])
    result_df = result_df.append(max_len_dic, ignore_index=True)

    #finds the min length value in the columns
    min_len_dic = dict( 
        [
            #create a tuple such that (column name, max length of values in column)
            (v, result_df[v].apply(lambda r: len(str(r))).min()) 
                for v in result_df.columns.values #iterates over all column values
        ])
    result_df = result_df.append(min_len_dic, ignore_index=True)

    #finds the average length for all the values inside the column
    mean_len_dic = dict( 
        [
            #create a tuple such that (column name, max length of values in column)
            (v, result_df[v].apply(lambda r: len(str(r))).mean()) 
                for v in result_df.columns.values #iterates over all column values
        ])
    result_df = result_df.append(mean_len_dic, ignore_index=True)

    #finds the data type of each column
    paga=result_df.dtypes
    ls_dt = []
    ls_name = []
    for name in result_df.columns:
        ls_name.append(name)
    for item in paga:
        ls_dt.append(str(item))
    dt_dic = {ls_name[i]: ls_dt[i] for i in range(len(ls_name))}
    # dt_type = {'int64': 0, 'float64': 1, 'object':2}
    for key,value in dt_dic.items():
        if value == 'int64':
            dt_dic[key] = 0
        elif value == 'float64':
            dt_dic[key] = 1
        else:
            dt_dic[key] = 2
    result_df = result_df.append(dt_dic, ignore_index=True)

    st.write(result_df.tail(12))

    ci = st.slider('Confidence Interaval (%)', 51, 99, 80)

    result_df = result_df.T

    data = result_df.to_numpy()
    X = data[:,-12:]
    x_name = data[:,-13]
    sc = StandardScaler()
    sc.fit(X)    
    X_std = sc.transform(X)

    load_clf = pickle.load(open('vandy_intern.pkl','rb'))
    p_array = load_clf.predict_proba(X_std)
    df = pd.DataFrame(p_array, index=x_name)
    fact_ls=[]
    dim_ls=[]
    uk_dic={}
    for row, col in df.iterrows():
        if col[0] >= (ci/100):
            fact_ls.append(row)
        elif col[0] <= (1 - (ci/100)):
            dim_ls.append(row)
        else:
            uk_dic[row] = [col[0],col[1]]
            
    col1, col2 = st.beta_columns(2)
    col1.header('Facts')
    col1.write(pd.DataFrame(fact_ls))

    col2.header('Dimensions')
    col2.write(pd.DataFrame(dim_ls))

    uk_df = pd.DataFrame(uk_dic).T
    uk_df.columns = ['Facts', 'Dimensions']
    st.write(uk_df)