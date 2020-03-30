from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import Counter
import pandas as pd
import numpy as np
import os

carnonical_codons= ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG', 'ATT', 'ATC',
                    'ATA', 'ATG', 'GTT', 'GTC', 'GTA', 'GTG', 'TCT', 'TCC', 'TCA', 'TCG',
                    'CCT', 'CCC', 'CCA', 'CCG', 'ACT', 'ACC', 'ACA', 'ACG', 'GCT', 'GCC',
                    'GCA', 'GCG', 'TAT', 'TAC', 'TAA', 'TAG', 'CAT', 'CAC', 'CAA', 'CAG',
                    'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA', 'GAG', 'TGT', 'TGC',
                    'TGA', 'TGG', 'CGT', 'CGC', 'CGA', 'CGG', 'AGT', 'AGC', 'AGA', 'AGG',
                    'GGT', 'GGC', 'GGA', 'GGG']
codons = []
for word in carnonical_codons:
    codons.append(word.lower())

def batch_maker(X, y, batch_size):
    rnd_indx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_indx in np.array_split(rnd_indx, n_batches):
        X_batch, y_batch = X[batch_indx], y[batch_indx]
        yield X_batch, y_batch


#Importing Data
path = os.path.join(os.getcwd(), 'Data', 'BOW_400.csv')
path_2 = os.path.join(os.getcwd(), 'Data', 'Seq_data.csv')

df = pd.read_csv(path, usecols=codons)
df2 = pd.read_csv(path_2, usecols=['Country'])
print(df.head(15))
print(df.shape)
print(df2.shape)
order_of_columns = list(df.columns)

#dealing with duplicate data
duplicate_indexes = list(df[df.duplicated()].index)
df.drop(index=duplicate_indexes, inplace=True)
df2.drop(index=duplicate_indexes, inplace=True)

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

#dataset partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True, random_state=42)
X_valid, X_train = X_train[:300], X_train[300:]
y_valid, y_train = y_train[:300], y_train[300:]
print(X_train.shape, X_test.shape)

knn_clf = KNeighborsClassifier()
score = cross_val_score(knn_clf, X_train, y_train, scoring='accuracy', cv=5)
print('Accuracy fo 5 fold CV =', score)
knn_clf.fit(X_train, y_train)
train_pred = knn_clf.predict(X_train)
y_pred = knn_clf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
bal_accuracy = balanced_accuracy_score(y_valid, y_pred)
print('Train accuracy =', accuracy_score(y_train, train_pred))
print('Validation accuracy =', accuracy)
print('Balanced validation accuracy =', bal_accuracy)

#prediction of coding sequences
CDS_BOW_path = os.path.join(os.getcwd(), 'Data_3', 'BOW_CDS.csv')
CDS_csv = os.path.join(os.getcwd(), 'Data_3', 'CDS.csv')
cds_lab = pd.read_csv(CDS_csv, usecols=['Country'])
cds_BW = pd.read_csv(CDS_BOW_path, usecols=codons)
print('Data for glycoproteinB coding sequences')
print(cds_BW.head())
print(cds_lab.head())
x_cds = scaler.fit_transform(cds_BW.values.astype(np.float32))
y_cds = encoder.fit_transform(cds_lab.values.ravel())

pred_cds = knn_clf.predict(x_cds)
acc_cds = accuracy_score(y_cds, pred_cds)
print(acc_cds)


