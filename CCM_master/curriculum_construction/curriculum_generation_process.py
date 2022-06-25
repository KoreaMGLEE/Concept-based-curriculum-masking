import os
import copy
import pickle

from curriculum_construction.curriculum_module import *


conceptnet_path = '/home/user3/Third_paper/data/bi_conceptnet'
with open(conceptnet_path, 'rb') as f:
    bi_conceptnet = pickle.load(f)

# load_first_curriculum
file_path = '/home/user3/data/junhyung_ex/curriculum_1k_1hop/'
first_curri_file = 'curriculum1'
with open(os.path.join(file_path, first_curri_file), 'rb') as f:
    curriculum1 = pickle.load(f)

# generate_second_curriculum
curriculum2 = generate_next_curriculum(bi_conceptnet, curriculum1, curriculum1)
all_previous_concepts = copy.deepcopy(curriculum1)
all_previous_concepts = add_new_concepts(all_previous_concepts, curriculum2)
with open(os.path.join(file_path, 'curriculum2'), 'wb') as f:
    pickle.dump(curriculum2, f)

curriculum3 = generate_next_curriculum(bi_conceptnet, curriculum2, all_previous_concepts)
all_previous_concepts = add_new_concepts(all_previous_concepts, curriculum3)
with open(os.path.join(file_path, 'curriculum2'), 'wb') as f:
    pickle.dump(curriculum3, f)


curriculum4 = generate_next_curriculum(bi_conceptnet, curriculum3, all_previous_concepts)
all_previous_concepts = add_new_concepts(all_previous_concepts, curriculum4)
with open(os.path.join(file_path, 'curriculum3'), 'wb') as f:
    pickle.dump(curriculum4, f)


for concept in curriculum4.keys():
    if concept in curriculum2.keys():
        print(concept)


with open('/home/user3/Third_paper/data/curriculum_basic_concept/curriculum3', 'rb') as f:
    final_concept = pickle.load(f)

last_concept_dic = {}
for concept in final_concept.keys():
    if concept not in all_previous_concepts.keys():
        last_concept_dic[concept] = 0

with open(os.path.join(file_path, 'curriculum3'), 'wb') as f:
    pickle.dump(last_concept_dic, f)

