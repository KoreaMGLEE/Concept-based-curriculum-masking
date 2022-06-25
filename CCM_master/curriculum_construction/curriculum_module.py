def crawling_concept_from_basicconcepts(concept_txt):
    basic_concepts = {}
    for line in concept_txt:
        if "//" in line:
            pass
        else:
            basic_concepts[line.replace('\n','')] = 0
    return basic_concepts


def generate_next_curriculum(bi_conceptnet, previous_curriculum, all_previous_curriculum):
    new_concept_dic = {}
    for key in previous_curriculum.keys():
        if key in bi_conceptnet.keys():
            for new_concept in bi_conceptnet[key].keys():
                if new_concept not in all_previous_curriculum.keys():
                    new_concept_dic[new_concept] = 0

    return new_concept_dic

import copy
def add_new_concepts(previous_curri, new_concepts):

    for key in new_concepts.keys():
        previous_curri[key] = 0
    return previous_curri