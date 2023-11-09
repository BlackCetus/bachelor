from Bio import SeqIO
from Bio import SwissProt
from Bio import ExPASy
import numpy as np
import pandas as pd
import requests as r
from io import StringIO


def add_to_fasta(fasta, cID, sequence):
    '''with a given fasta file, adds the given sequence of a given cID to the file'''
    with open(fasta, 'a') as file:
        file.write('>'+cID+'\n')
        file.write(str(sequence)+'\n')


def remove_second_column(file):
    df = pd.read_csv(file)
    df = df.drop(df.columns[0], axis=1)
    df.to_csv(file, index=False)   
    

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
    # gets seq from fasta, if not available, takes from uniprot website
    seq_a = []
    seq_b = []
    drop = []
    for index, row in df.iterrows():
        id1 = row['Id1']
        id2 = row['Id2']
        if id1 in fasta_df["ID"].values:
            a = str(fasta_df[fasta_df["ID"] == id1]["Sequence"].values[0])
        else:   
            a = get_seq(id1)
            add_to_fasta('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta', id1, a)
            fasta_df = read_fasta_as_df('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta')
            print(id1)
        if id2 in fasta_df["ID"].values:
            b = str(fasta_df[fasta_df["ID"] == id2]["Sequence"].values[0])
        else:   
            b = get_seq(id2) 
            add_to_fasta('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta', id2, b)
            fasta_df = read_fasta_as_df('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta')
            print(id2)
        #if len(a) <= 1166 and len(b) <= 1166:
        seq_a.append(a)
        seq_b.append(b)
        #else:
        #    drop.append(index)          
    #df = df.drop(drop)          
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


# get fasta
fasta_df = read_fasta_as_df('bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta')


# get data
train_pos = file2df("bachelor/pytorchtest/data/gold_stand/Intra1_pos_rr.txt", 1)
train_neg = file2df("bachelor/pytorchtest/data/gold_stand/Intra1_neg_rr.txt", 0)
test_pos = file2df("bachelor/pytorchtest/data/gold_stand/Intra2_pos_rr.txt", 1)
test_neg = file2df("bachelor/pytorchtest/data/gold_stand/Intra2_neg_rr.txt", 0)
val_pos = file2df("bachelor/pytorchtest/data/gold_stand/Intra0_pos_rr.txt", 1)
val_neg = file2df("bachelor/pytorchtest/data/gold_stand/Intra0_neg_rr.txt", 0)

# join neg and pos
train_all = pd.concat([train_pos, train_neg]).reset_index()
test_all = pd.concat([test_pos, test_neg]).reset_index()
val_all = pd.concat([val_pos, val_neg]).reset_index()

# add sequence
train_all_seq = addseqtodf_ff(train_all, fasta_df)
test_all_seq = addseqtodf_ff(test_all, fasta_df)
val_all_seq = addseqtodf_ff(val_all, fasta_df)

# save to csv
train_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand/train_intra1_all_seq.csv')
test_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand/test_intra2_all_seq.csv')
val_all_seq.to_csv('bachelor/pytorchtest/data/gold_stand/val_intra0_all_seq.csv')


# remove second column
remove_second_column('bachelor/pytorchtest/data/gold_stand/train_intra1_seq.csv')
remove_second_column('bachelor/pytorchtest/data/gold_stand/test_intra2_seq.csv')
remove_second_column('bachelor/pytorchtest/data/gold_stand/val_intra0_seq.csv')
