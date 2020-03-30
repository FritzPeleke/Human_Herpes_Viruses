import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from time import time
import seaborn as sns
pd.options.display.width = 0

def reset_graph():
    tf.reset_default_graph()
    np.random.seed(seed=42)
    tf.set_random_seed(seed=42)


reset_graph()
#The 64 carnonical codons which code for amino acids
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
df2.rename(columns ={'Country':'Region'}, inplace=True)
print(df.head(15))
print(df.shape)
print(df2.shape)
order_of_columns = list(df.columns)

#dealing with duplicate data
duplicate_indexes = list(df[df.duplicated()].index)
df.drop(index=duplicate_indexes, inplace=True)
df2.drop(index=duplicate_indexes, inplace=True)
sns.countplot(y='Region', data=df2)


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

n_inputs = X.shape[1]
hidden1 = 60
hidden2 = 40
hidden3 = 40
hidden4 = 40
n_outputs = 5

#construction phase
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

He_init = tf.variance_scaling_initializer()
with tf.name_scope('DNN'):
    hid_layer_1 = tf.layers.dense(X, hidden1, activation=tf.nn.relu,
                                  name='Hidden1')
    hid_layer_2 = tf.layers.dense(hid_layer_1, hidden2, activation=tf.nn.relu,
                                  name='Hidden2')
    dropout_layer_2 = tf.layers.dropout(hid_layer_2, rate=0.75, training=training, name='dropout')
    hid_layer_3 = tf.layers.dense(dropout_layer_2, hidden3, activation=tf.nn.relu,
                                  name='Hidden3')
    hid_layer_4 = tf.layers.dense(hid_layer_3, hidden4, activation=tf.nn.relu,
                                  name='Hidden4')
    logits = tf.layers.dense(hid_layer_4, n_outputs, name='outputs')

with tf.name_scope('loss'):
    Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(Xentropy, name='loss')
    loss_sum = tf.summary.scalar('loss_sumr', loss)

with tf.name_scope('train'):# Applying learning schedule
    initial_learn_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learn_rate = tf.train.exponential_decay(initial_learn_rate, global_step, decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    acc_smr = tf.summary.scalar('acc_smr', accuracy)

n_epochs = 10000
batch_size = 100
init = tf.global_variables_initializer()
best_loss = np.infty
check_without_progress = 0
max_epoch_without_imporvement = 40
saver = tf.train.Saver()
logdir = os.path.join(os.getcwd(), 'FNW_logs', 'FNW{}'.format(time()))
filewriter = tf.summary.FileWriter(logdir, tf.get_default_graph())
#Execution phase
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for x_batch, y_batch in batch_maker(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X:  x_batch, y: y_batch})
        acc_value, loss_value, loss_summary, acc_summary = sess.run([accuracy, loss, loss_sum, acc_smr],
                                                                    feed_dict={X: X_valid, y: y_valid})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        filewriter.add_summary(acc_summary, epoch)
        filewriter.add_summary(loss_summary, epoch)

        if loss_value < best_loss: #implementing early_stopping
            best_loss = loss_value
            check_without_progress = 0
        else:
            check_without_progress += 1
            if check_without_progress > max_epoch_without_imporvement:
                print('Early_stopping')
                break
        print('Epoch:', epoch, 'Loss:', loss_value, '\tbest_loss', best_loss, '\tval_acc:', acc_value, 'train_acc:',
              acc_train)
    saver.save(sess, '.\\my_sequenz_model.ckpt')


#Testing coding sequences
CDS_BOW_path = os.path.join(os.getcwd(), 'Data_3', 'BOW_CDS.csv')
CDS_csv = os.path.join(os.getcwd(), 'Data_3', 'CDS.csv')
cds_lab = pd.read_csv(CDS_csv, usecols=['Country'])
cds_lab.rename(columns ={'Country':'Region'}, inplace=True)
cds_BW = pd.read_csv(CDS_BOW_path, usecols=codons)
print('Data for glycoproteinB coding sequences')
print(cds_BW.head())
print(cds_lab.head())
x_cds = scaler.fit_transform(cds_BW.values.astype(np.float32))
y_cds = encoder.fit_transform(cds_lab.values.ravel())

with tf.Session() as sess:
    saver.restore(sess, '.\\my_sequenz_model.ckpt')
    z = logits.eval(feed_dict={X: X_valid})
    val_acc = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
    train_acc = accuracy.eval(feed_dict={X: X_train, y: y_train})
    cds_acc = accuracy.eval(feed_dict={X: x_cds, y: y_cds})
    z_cds = logits.eval(feed_dict={X: x_cds})
    cds_pred = np.argmax(z_cds, axis=1) #predictions for coding sequences

    y_pred = np.argmax(z, axis=1)#predictions for validation set
    print('validation_accuracy: ', val_acc)
    print('training_accuracy :', train_acc)
    print('CDS_accuracy: ', cds_acc)

#Exploiting predictions with sklearn
con_mat = confusion_matrix(y_valid, y_pred)
print(con_mat)
print('Validation_accuracy:', accuracy_score(y_valid, y_pred))
print('Balanced_val_accuracy', balanced_accuracy_score(y_valid, y_pred))

#Represnting the errors in the confusion matrix on an image
row_sums= con_mat.sum(axis=1, keepdims=True)
norm_conf_mx = con_mat / row_sums
print(norm_conf_mx)
np.fill_diagonal(norm_conf_mx, 0)
fig1, ax1 = plt.subplots()
ax1.matshow(norm_conf_mx, cmap=plt.cm.gray)
im1 = ax1.matshow(norm_conf_mx, cmap=plt.cm.gray)
ax1.set_xlabel('Predicted classes')
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Actual classes')
fig1.colorbar(im1)
plt.show()

#confusion matrix for coding sequences
print('Confusion matrix for 13 coding sequences')
matrix = confusion_matrix(y_cds, cds_pred)
print(matrix)
fig2, ax2 = plt.subplots()
ax2.matshow(matrix, cmap=plt.cm.gray)
im2 = ax2.matshow(matrix, cmap=plt.cm.gray)
ax2.set_xlabel('Predicted classes')
ax2.xaxis.set_label_position('top')
ax2.set_ylabel('Actual classes')
fig2.colorbar(im2)
plt.show()