import pandas as pd
from Bio import SeqIO
import os
import numpy as np
import random

pd.options.display.width = 0
root_dir = os.path.join(os.getcwd(), 'Data')
seq_csv_path = root_dir + '\\sequences.csv'
seq_fasta_path = root_dir + '\\sequences.fasta'
df = pd.read_csv(seq_csv_path, usecols=['Accession', 'Species', 'Sequence_Type', 'Country',
                                        'Host',  'Length'])


#Creating labels
df.replace(['Czech Republic','France', 'Italy', 'Germany', 'Netherlands', 'Belgium', 'Poland', 'Ireland', 'Russia',
            'Ukraine', 'Switzerland','Sweden','France','Denmark', 'Greece','United Kingdom', 'Zaire', 'Zambia', 'Kenya',
            'Zimbabwe', 'Cameroon', 'Gambia', 'Democratic Republic of the Congo', 'Morocco', 'South Africa', 'Uganda',
            'Ghana', 'Egypt', 'Nigeria', 'Tanzania', 'Peru', 'Brazil', 'USA', 'Argentina', 'Mexico','Haiti', 'Australia',
            'Wallis and Futuna','New Zealand', 'Iran', 'Japan', 'Taiwan', 'Hong Kong', 'India', 'China', 'Israel',
            'Saudi Arabia', 'South Korea'],
           ['Europe','Europe', 'Europe', 'Europe', 'Europe', 'Europe','Europe','Europe','Europe','Europe',
            'Europe','Europe','Europe','Europe','Europe', 'Europe','Africa', 'Africa', 'Africa', 'Africa', 'Africa',
            'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'America',
            'America','America','America','America','America', 'Oceania', 'Oceania', 'Oceania', 'Asia', 'Asia', 'Asia',
            'Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'Asia'],
           inplace=True)

list_of_records_with_more_than_600bp = []
for record in SeqIO.parse(seq_fasta_path, 'fasta'):
    if len(record.seq) > 600:
        list_of_records_with_more_than_600bp.append(record)


write_path = root_dir + '\\Seq_greater_600bp.fasta'
#Writing sequences with more than 600bp
SeqIO.write(list_of_records_with_more_than_600bp, write_path, 'fasta')

Accesion_numbers = []
Length = []
  
for rec in SeqIO.parse(write_path, 'fasta'):
    Accesion_numbers.append(rec.description.split(' ')[0])
    Length.append(len(rec.seq))

original_accessions = []
for acc in df.Accession:
    original_accessions.append(acc)


index = [original_accessions.index(ID) for ID in Accesion_numbers]
df_to_array = df.values # getting the sequences by index
sequence_arrays = df_to_array[index]
Final_dataframe = pd.DataFrame(sequence_arrays, columns=['Accession', 'Species', 'Sequence_Type', 'Country', 
                                                         'Host',  'Length'])


Final_csv_path = root_dir + '\\Seq_data.csv'
# Saving dataframe
Final_dataframe.to_csv(Final_csv_path, columns=['Accession', 'Species', 'Sequence_Type', 'Country',
                                                'Host', 'Length'], index=False)



def cutout_maker(target_list):
    Length = 400
    result = []
    if len(target_list) < Length:
        result.append(target_list(target_list[:]))
    else:
        for i in range(len(target_list) - Length +1):
            result.append(target_list[i:i + Length])
    choice = random.choice(result)
    return choice


List_of_cutouts = []
for record in SeqIO.parse(write_path, 'fasta'):
    rec_seq = list(record.seq)
    cut_out = cutout_maker(rec_seq)
    List_of_cutouts.append(list(cut_out))


data_path = root_dir + '\\sequences_400_cutouts.csv'
dataframe = pd.DataFrame(List_of_cutouts)
dataframe.to_csv(data_path, index=False)





