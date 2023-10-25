from Bio import SeqIO
from Bio import SwissProt
from Bio import ExPASy
import numpy as np
import pandas as pd
import requests as r
from io import StringIO


def file2df(file_path, interact):

    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['Id1', 'Id2'])
    df['Interact'] = interact
    return df

def get_seq(cID):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq, 'fasta'))
    return pSeq[0].seq

def addseqtodf(df):
    seq_a = []
    seq_b = []
    for index, row in df.iterrows():
        a = get_seq(row['Id1'])
        b = get_seq(row['Id2'])
        if len(a) <= 1166 and len(b) <=1166:
            seq_a.append(a)
            seq_b.append(b)
    df['sequence_a'] = seq_a
    df['sequence_b'] = seq_b
    return df

def addseqtodf_ff(df, fasta_df):
    seq_a = []
    seq_b = []
    drop = []
    for index, row in df.iterrows():
        id1 = row['Id1']
        id2 = row['Id2']
        a = str(fasta_df[fasta_df["ID"] == id1]["Sequence"].values[0])
        b = str(fasta_df[fasta_df["ID"] == id2]["Sequence"].values[0])
        if len(a) <= 1166 and len(b) <= 1166:
            seq_a.append(a)
            seq_b.append(b)
        else:
            drop.append(index) 
    print(len(df))
    print(len(drop))          
    df = df.drop(drop)    
    print(len(df))        
    df['sequence_a'] = seq_a
    df['sequence_b'] = seq_b
    return df

def read_fasta_as_df(fasta):
    with open(fasta) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    seqs = [x for x in content if x[0] != '>']
    ids = [x.split('>')[1] for x in content if x[0] == '>']
    df = pd.DataFrame()
    df["ID"] = ids
    df["Sequence"] = seqs
    return df


fasta_df = read_fasta_as_df('bachelor/pytorchtest/data/human_swissprot_oneliner.fasta')


print(fasta_df)

train_pos = file2df("bachelor/pytorchtest/data/huang_train_pos.txt", 1)
train_neg = file2df("bachelor/pytorchtest/data/huang_train_neg.txt", 0)
test_pos = file2df("bachelor/pytorchtest/data/huang_test_pos.txt", 1)
test_neg = file2df("bachelor/pytorchtest/data/huang_test_pos.txt", 0)

print(1)
print(train_pos)

train_all = pd.concat([train_pos, train_neg]).reset_index()
test_all = pd.concat([test_pos, test_neg]).reset_index()

print(2)
print(train_all)


#train_all_seq = addseqtodf(train_all)
#test_all_seq = addseqtodf(test_all)

train_all_seq = addseqtodf_ff(train_all, fasta_df)
test_all_seq = addseqtodf_ff(test_all, fasta_df)

train_all_seq.to_csv('bachelor/pytorchtest/data/train_all_seq_1166.csv')
test_all_seq.to_csv('bachelor/pytorchtest/data/test_all_seq_1166.csv')

