from Bio import  SeqIO
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
pd.options.display.width = 0
CDS_path = os.path.join(os.getcwd(), 'Data_3', 'non_redundant.fasta')
seq_csv = os.path.join(os.getcwd(), 'Data_2', 'sequences.csv')
CDS_csv = os.path.join(os.getcwd(), 'Data_3', 'CDS.csv')
CDS_BOW_path = os.path.join(os.getcwd(), 'Data_3', 'BOW_CDS.csv')

df = pd.read_csv(seq_csv)
columns = ['Accession', 'Species', 'Nuc._Completeness', 'Country', 'Host',
           'Length']
Accession_numbers = [acc_num for acc_num in df.Accession]
ID = []
for record in SeqIO.parse(CDS_path, 'fasta'):
    ID.append(record.id.split('.')[0])
df_to_array = df.values
accession_num_indexes = [Accession_numbers.index(identifier) for identifier in ID]
df_2 = pd.DataFrame(df_to_array[accession_num_indexes], columns=columns)

df_2.replace(['Czech Republic', 'Turkey', 'France', 'Italy', 'Germany', 'Netherlands', 'Belgium', 'Poland', 'Ireland', 'Russia',
            'Ukraine', 'Switzerland','Sweden','France','Denmark', 'Greece','United Kingdom', 'Zaire', 'Zambia', 'Kenya',
            'Zimbabwe', 'Cameroon', 'Gambia', 'Democratic Republic of the Congo', 'Morocco', 'South Africa', 'Uganda',
            'Ghana', 'Egypt', 'Nigeria', 'Tanzania', 'Peru', 'Brazil', 'USA', 'Cuba', 'Argentina', 'Mexico','Haiti', 'Australia',
            'Wallis and Futuna','New Zealand', 'Iran', 'Japan', 'Taiwan', 'Hong Kong', 'India', 'China', 'Israel',
            'Saudi Arabia', 'South Korea','Iraq'],
           ['Europe','Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe','Europe','Europe','Europe','Europe',
            'Europe','Europe','Europe','Europe','Europe', 'Europe','Africa', 'Africa', 'Africa', 'Africa', 'Africa',
            'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'America',
            'America','America','America', 'America', 'America','America', 'Oceania', 'Oceania', 'Oceania', 'Asia', 'Asia', 'Asia',
            'Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'Asia'],
           inplace=True)
df_2.to_csv(CDS_csv, index=False)
#Generating the bag of words and finally histogram
ls_of_words = []
for rec in SeqIO.parse(CDS_path, 'fasta'):
    record = list(rec.seq)
    joint_seq = ''.join(record)
    n = 3
    triplets = [joint_seq[i:i + n] for i in range(0, len(joint_seq), n)]
    ls_of_words.append(triplets)
ls_of_corpus = []
for triplets_list in ls_of_words:
    corpus = ' '.join(str(item) for item in triplets_list)
    ls_of_corpus.append(corpus)
vectorizer = CountVectorizer()
histogram = vectorizer.fit_transform(ls_of_corpus).todense()
CDS_BOW = pd.DataFrame(data=histogram, columns=vectorizer.get_feature_names())
CDS_BOW.to_csv(CDS_BOW_path, index=False)

