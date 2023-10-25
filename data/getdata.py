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
        seq_a.append(get_seq(row['Id1']))
        seq_b.append(get_seq(row['Id2']))
    df['sequence_a'] = seq_a
    df['sequence_b'] = seq_b
    return df



train_pos = file2df("bachelor/pytorchtest/data/huang_train_pos.txt", 1)
train_neg = file2df("bachelor/pytorchtest/data/huang_train_neg.txt", 0)
test_pos = file2df("bachelor/pytorchtest/data/huang_test_pos.txt", 1)
test_neg = file2df("bachelor/pytorchtest/data/huang_test_pos.txt", 0)

print(1)
print(train_pos)

train_all = pd.concat([train_pos, train_neg])
test_all = pd.concat([test_pos, test_neg])

print(2)
print(train_all)


train_all_seq = addseqtodf(train_all)
test_all_seq = addseqtodf(test_all)

train_all_seq.to_csv('bachelor/pytorchtest/data/train_all_seq.csv')
test_all_seq.to_csv('bachelor/pytorchtest/data/test_all_seq.csv')

