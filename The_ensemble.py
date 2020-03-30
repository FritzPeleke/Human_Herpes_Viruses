from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score
from dask_ml.model_selection import GridSearchCV
from sklearn.base import clone
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import os
from collections import  Counter
pd.options.display.width = 0
np.random.seed(seed=42)

#The correct 64 codons which code for amino acids
possible_codons= ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG', 'ATT', 'ATC',
                  'ATA', 'ATG', 'GTT', 'GTC', 'GTA', 'GTG', 'TCT', 'TCC', 'TCA', 'TCG',
                  'CCT', 'CCC', 'CCA', 'CCG', 'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC',
                  'GCA', 'GCG', 'TAT', 'TAC', 'TAA', 'TAG', 'CAT', 'CAC', 'CAA', 'CAG',
                  'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA', 'GAG', 'TGT', 'TGC',
                  'TGA', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG', 'AGT', 'AGC', 'AGA', 'AGG',
                  'GGT', 'GGC', 'GGA', 'GGG']
codons = []
for word in possible_codons:
    codons.append(word.lower())

#Importing Data
path = os.path.join('C:\\Users\\fritz\\Tensorflow', 'Data', 'BOW_400.csv')
path_2 = os.path.join('C:\\Users\\fritz\\Tensorflow', 'Data', 'Seq_data.csv')

df = pd.read_csv(path, usecols= codons)
df2 = pd.read_csv(path_2, usecols=['Country'])
print(df.head(15))
print(df.shape)
print(df2.shape)

#Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X = df.values.astype(np.float32)
X = scaler.fit_transform(X)
y = df2.values
print('Counter for general dataset: \n', Counter(y.ravel()).items())
#Label encoding
encoder = LabelEncoder()
encoder.fit(y.ravel())
classes = encoder.classes_
y_encoded = encoder.transform(y.ravel()).astype(np.int32)

print(classes)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True, random_state=42)
X_valid, X_train = X_train[:300], X_train[300:]
y_valid, y_train = y_train[:300], y_train[300:]

GB_clf = GradientBoostingClassifier(n_estimators=350)
Ad_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=350)
xg_clf = XGBClassifier(n_estimators=200)
vot_clf = VotingClassifier([('GB_clf', GB_clf),
                            ('xg_clf', xg_clf),
                            ('ada_clf', Ad_clf)])

vot_clf.fit(X_train, y_train)
y_pred = vot_clf.predict(X_valid)
train_pred = vot_clf.predict(X_train)
print('Validation_accuracy =', accuracy_score(y_valid, y_pred))
print('Train_accuracy =', accuracy_score(y_train, train_pred))
print('Balanced_val_accuracy =', balanced_accuracy_score(y_valid, y_pred))
test_pred = vot_clf.predict(X_test)
print('Test_accuracy:', accuracy_score(y_test, test_pred))

#Prediction for  the coding sequence
CDS_BOW_path = os.path.join('C:\\Users\\fritz\\Tensorflow', 'Data_3', 'BOW_CDS.csv')
CDS_csv = os.path.join('C:\\Users\\fritz\\Tensorflow', 'Data_3', 'CDS.csv')
cds_lab = pd.read_csv(CDS_csv, usecols=['Country'])
cds_BW = pd.read_csv(CDS_BOW_path, usecols=codons)
print('Data for glycoproteinB coding sequences')
print(cds_BW.head())
print(cds_lab.head())
x_cds = scaler.fit_transform(cds_BW.values.astype(np.float32))
y_cds = encoder.fit_transform(cds_lab.values.ravel())

pred_cds = vot_clf.predict(x_cds)
acc_cds = accuracy_score(y_cds, pred_cds)
print('Acuracy coding seq: ', acc_cds)