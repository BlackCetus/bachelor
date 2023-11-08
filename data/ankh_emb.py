import ankh
import torch
from Bio import SeqIO
import os

# maybe use later

file_path = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner.fasta"

model, tokenizer = ankh.load_large_model()
model.eval()
protein_sequences = ['MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQSVTSRGDLATCPRGFAVTGCTCGSACGSWDVRAETTCHCQCAGMDWTGARCCRVQPLEHHHHHH', 
'GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEEMRKEIIAVIQKYTKSSDIHFKTLDSNQSVETIEVEIILPR']

protein_sequences = [list(seq) for seq in protein_sequences]

print(2)
outputs = tokenizer.batch_encode_plus(protein_sequences, 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
print(3)                                    
with torch.no_grad():
    embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

test_to_keep_going = 1