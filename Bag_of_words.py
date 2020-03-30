import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
pd.options.display.width = 0

path_to_data = os.path.join(os.getcwd(), 'Data', 'sequences_400_cutouts.csv')
path_to_label = os.path.join(os.getcwd(), 'Data', 'Seq_data.csv' )
save_path = os.path.join(os.getcwd(), 'Data', 'BOW_400.csv')
data = pd.read_csv(path_to_data)
labels = pd.read_csv(path_to_label, usecols=['Country'])

#converting data to array and generating new data
data_array = data.values
ls = []
for element in data_array:
    joint_seq = ''.join(element)
    n = 3
    triplets = [joint_seq[i:i + n] for i in range(0, len(joint_seq), n)]
    ls.append(triplets)

new_ls = []
for ls_item in ls:
    corpus = ' '.join(str(item) for item in ls_item)
    new_ls.append(corpus)
vectorizer = CountVectorizer()
histogram = vectorizer.fit_transform(new_ls).todense()
BOW_df = pd.DataFrame(data=histogram, columns=vectorizer.get_feature_names())

#writing BOW into a csv file
BOW_df.to_csv(save_path, index=False)
