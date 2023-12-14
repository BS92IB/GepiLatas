import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn as sk
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter





#model = keras.models.load_model('./OverSampleModel')

#model_1 = keras.models.load_model('./UnderSampleModel')


def main():

    model = keras.models.load_model('OverSampleModel')
    model_1 = keras.models.load_model('UnderSampleModel')

    st.title("Gépi Tanulás - Credit Card Fraud Detection App - BS92IB")

    with open('UnderSampleLogReg.pkl', 'rb') as file:
        model_2 = pickle.load(file)
    with open('OverSampleLogReg.pkl', 'rb') as file1:
        model_3 = pickle.load(file1)

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    


    df = pd.read_pickle('your_dataframe.pkl')


    X = df.drop('Class', axis=1)
    y = df['Class']

    

    for train_index, test_index in sss.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
    # original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the Distribution of the labels


    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    undersample_X = df.drop('Class', axis=1)
    undersample_y = df['Class']

    for train_index, test_index in sss.split(undersample_X, undersample_y):
        print("Train:", train_index, "Test:", test_index)
        undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
        undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
    undersample_Xtrain = undersample_Xtrain.values
    undersample_Xtest = undersample_Xtest.values
    undersample_ytrain = undersample_ytrain.values
    undersample_ytest = undersample_ytest.values

    undersample_accuracy = []
    undersample_precision = []
    undersample_recall = []
    undersample_f1 = []
    undersample_auc = []

    # Implementing NearMiss Technique 
    # Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
    nearmiss = NearMiss()
    X_nearmiss, y_nearmiss = nearmiss.fit_resample(undersample_X.values, undersample_y.values)
    print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))
    # Cross Validating the right way

    log_reg = LogisticRegression()

    for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
        undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
        undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
        undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
        
        undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
        undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
        undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
        undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
        undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))




    


    oversample_log_reg = model_3.predict(original_Xtest)
    class_report_3 = classification_report(original_ytest, oversample_log_reg, output_dict=True)

    st.subheader("Oversample Logistic Regression Results")

    st.table(class_report_3)
    accuracy = class_report_3['accuracy']
    precision_0 = class_report_3['0']['precision']
    precision_1 = class_report_3['1']['precision']
    recall_0 = class_report_3['0']['recall']
    recall_1 = class_report_3['1']['recall']

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision (Class 0): {precision_0:.2f}")
    st.write(f"Precision (Class 1): {precision_1:.2f}")
    st.write(f"Recall (Class 0): {recall_0:.2f}")
    st.write(f"Recall (Class 1): {recall_1:.2f}")

    
    # undersample_log_reg = model_2.predict(undersample_Xtest)
    #class_report_2 = classification_report(undersample_ytest, undersample_log_reg, output_dict=True)

    #st.subheader("Undersample Logistic Regression Results")

    #st.table(class_report_2)
    #accuracy = class_report_2['accuracy']
    #precision_0 = class_report_2['0']['precision']
    #precision_1 = class_report_2['1']['precision']
    #recall_0 = class_report_2['0']['recall']
    #recall_1 = class_report_2['1']['recall']

    #st.write(f"Accuracy: {accuracy:.2f}")
    #st.write(f"Precision (Class 0): {precision_0:.2f}")
    #st.write(f"Precision (Class 1): {precision_1:.2f}")
    #st.write(f"Recall (Class 0): {recall_0:.2f}")
    #st.write(f"Recall (Class 1): {recall_1:.2f}") 

    

    oversample_fraud_predictions = model.predict(original_Xtest, batch_size=200, verbose=0)

    threshold = 0.5
    oversample_fraud_predictions_binary = (oversample_fraud_predictions[:, 1] > threshold).astype(int)

    st.subheader("Oversample Neural Network Results")

    class_report = classification_report(original_ytest, oversample_fraud_predictions_binary, output_dict=True)

    st.table(class_report)
    accuracy = class_report['accuracy']
    precision_0 = class_report['0']['precision']
    precision_1 = class_report['1']['precision']
    recall_0 = class_report['0']['recall']
    recall_1 = class_report['1']['recall']

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision (Class 0): {precision_0:.2f}")
    st.write(f"Precision (Class 1): {precision_1:.2f}")
    st.write(f"Recall (Class 0): {recall_0:.2f}")
    st.write(f"Recall (Class 1): {recall_1:.2f}")

    undersample_fraud_predictions = model_1.predict(original_Xtest, batch_size=200, verbose=0)

    threshold = 0.5
    undersample_fraud_predictions_binary = (undersample_fraud_predictions[:, 1] > threshold).astype(int)

    st.subheader("Undersample Neural Network Results")

    class_report_1 = classification_report(original_ytest, undersample_fraud_predictions_binary, output_dict=True)

    st.table(class_report_1)
    accuracy = class_report_1['accuracy']
    precision_0 = class_report_1['0']['precision']
    precision_1 = class_report_1['1']['precision']
    recall_0 = class_report_1['0']['recall']
    recall_1 = class_report_1['1']['recall']

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision (Class 0): {precision_0:.2f}")
    st.write(f"Precision (Class 1): {precision_1:.2f}")
    st.write(f"Recall (Class 0): {recall_0:.2f}")
    st.write(f"Recall (Class 1): {recall_1:.2f}")


if __name__ == "__main__":
    main()