import pandas as pd
import numpy as np
from Bio import SeqIO, Entrez
import os

read_path = os.path.join(os.getcwd(), 'Data_2', 'sequences.csv')
write_path = os.path.join(os.getcwd(), 'Data_2', 'database.gb')
df = pd.read_csv(read_path)
list_of_accession_numbers = []
for accession in df.Accession:
    list_of_accession_numbers.append(accession)

Entrez.email = 'fpeleke@yahoo.com'
record_list = []
with Entrez.efetch(db='nucleotide', rettype='gb', retmode='text', id=list_of_accession_numbers) as handle:
    for record in SeqIO.parse(handle, 'gb'):
        record_list.append(record)
#writing records
SeqIO.write(record_list, write_path, 'gb')

