import csv
import pickle
from time import time
# read csv file
csv_file = open("../data/assertions.csv", "r", encoding='utf-8')
f = csv.reader(csv_file, delimiter="\t")

## extract triplet
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def add_Concept_to_dictionary(dictionary, head, tail):
    ## add head to tail
    if head not in dictionary.keys():
        dictionary[head] = {tail: True}
    else:
        tmp = dictionary[head]
        tmp[tail] = True
        dictionary[head] = tmp

    ## add tail to head
    if tail not in dictionary.keys():
        dictionary[tail] = {head: True}
    else:
        tmp = dictionary[tail]
        tmp[head] = True
        dictionary[tail] = tmp

    return dictionary

def extract_triplet(file):
    bi_conceptnet = {}
    for i, row in enumerate(f):
        head_node = row[2].split('/')[-2]
        if isEnglish(head_node) == False:
            continue
        tail_node = row[3].split('/')[-1]
        if isEnglish(tail_node) == False:
            continue
        relation = row[1].split('/')[-1]
        if isEnglish(relation) == False:
            continue

        bi_conceptnet = add_Concept_to_dictionary(bi_conceptnet, head_node, tail_node)
        # triplet = (head_node, tail_node, relation)
        if i > 2000:
            break
    return bi_conceptnet

start_time = time()
conceptnet = extract_triplet(f)

print(f"finish pre-processing: {time()-start_time}")

with open('../data/preprocessed_conceptnet', 'wb') as f:
    pickle.dump(conceptnet, f)