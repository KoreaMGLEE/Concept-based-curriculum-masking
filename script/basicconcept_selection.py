import os
import re
import pickle
from src.module import count_concept_
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conceptnet_path", default='../data/preprocessed_conceptnet', type=str)
parser.add_argument("--topk_connected_concepts", default=5000, type=int)
parser.add_argument("--corpus_dir", default='../data/pre-training-corpus/', type=str)
parser.add_argument("--delete_threshold", default=100000, type=int)
parser.add_argument("--basicConcepts_num", default=3000, type=int)
parser.add_argument("--save_path", default='../data/basic_concepts', type=str)
args = parser.parse_args()

with open(args.conceptnet_path, 'rb') as f:
    conceptnet = pickle.load(f)

## select top k concepts based on the number of connected edges
edge_count = {}
for key in conceptnet.keys():
    edge_count[key] = len(conceptnet[key])

topk = args.topk_connected_concepts
topk_connected_concepts = {k.replace('_', ' '): v for k, v in sorted(edge_count.items(), key=lambda item: item[1])[-topk:]}


## count frequency in the corpus
corpus_dir = args.corpus_dir
ConceptCount = {}
for file in os.listdir(corpus_dir):
    with open(os.path.join(corpus_dir, file), 'r') as f:
        lines = f.readlines()
    ConceptCount = count_concept_(lines, topk_connected_concepts, ConceptCount)


## delete concepts occurring less than 100k times
delete_threshold = args.delete_threshold

intermediate_conceptsSet = {}
for key in ConceptCount.keys():
    if ConceptCount[key] > delete_threshold:
        intermediate_conceptsSet[key] = topk_connected_concepts[key]

number_of_basicConcepts = args.basicConcepts_num
basic_concepts = {k.replace('_', ' '): v for k, v in sorted(intermediate_conceptsSet.items(), key=lambda item: item[1])[-number_of_basicConcepts:]}


with open(args.save_path, 'wb') as f:
    pickle.dump(basic_concepts, f)

