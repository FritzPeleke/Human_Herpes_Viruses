import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
counter = 0
CDS_records = []
database_path = os.path.join(os.getcwd(), 'Data_2', 'database.gb')
write_path = os.path.join(os.getcwd(), 'Data_3', 'MCP_database.fasta')
for record in SeqIO.parse(database_path, 'gb'):
    for feature in record.features:
        if feature.type == 'CDS':
            for qualifiers in feature.qualifiers:
                if qualifiers == 'product':
                    for product in feature.qualifiers['product']:
                        if feature.qualifiers['product'] == ['Major capsid protein']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['major capsid protein']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['UL19']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                description=record.description)
                            counter += 1

                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['40']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['UL86']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['U57']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                 description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['BcLF1']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                 description=record.description)
                            counter += 1
                            CDS_records.append(sequence)
                        elif feature.qualifiers['product'] == ['ORF25']:
                            CDS = feature.location.extract(record)
                            sequence = SeqRecord(CDS.seq, id=record.id,
                                                 description=record.description)
                            counter += 1
                            CDS_records.append(sequence)



SeqIO.write(CDS_records, write_path, 'fasta')
print(counter)
