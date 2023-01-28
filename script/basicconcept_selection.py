import os
import re
import pickle
from src.module import split_text_ngram, count_concept_


with open('../data/preprocessed_conceptnet', 'rb') as f:
    conceptnet = pickle.load(f)


## select top k concepts based on the number of connected edges
edge_count = {}
for key in conceptnet.keys():
    edge_count[key] = len(conceptnet[key])

topk = 5000
topk_connected_concepts = {k.replace('_', ' '): v for k, v in sorted(edge_count.items(), key=lambda item: item[1])[-topk:]}


## count frequency in the corpus
corpus_dir = '../data/pre-training-corpus/'
ConceptCount = {}
for file in os.listdir(corpus_dir):
    with open(os.path.join(corpus_dir, file), 'r') as f:
        lines = f.readlines()
    ConceptCount = count_concept_(lines, topk_connected_concepts, ConceptCount)


## delete concepts occurring less than 100k times
delete_threshold = 100000

intermediate_conceptsSet = {}
for key in ConceptCount.keys():
    if ConceptCount[key] > delete_threshold:
        intermediate_conceptsSet[key] = topk_connected_concepts[key]

number_of_basicConcepts = 3000
basic_concepts = {k.replace('_', ' '): v for k, v in sorted(intermediate_conceptsSet.items(), key=lambda item: item[1])[-number_of_basicConcepts:]}


with open('../data/concept_based_curriculum/basic_concepts', 'wb') as f:
    pickle.dump(basic_concepts, f)

