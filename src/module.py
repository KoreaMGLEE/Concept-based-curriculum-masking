def split_text_ngram(text):
    word_list = text.split()
    word_list_len = len(word_list)
    multigram_cpt_list = []
    bigram_cpt_list = []
    trigram_cpt_list = []
    fourgram_cpt_list = []

    for bigram_idx in range(word_list_len - 1):
        bi_list = word_list[bigram_idx:2 + bigram_idx]
        bigram_cpt = bi_list[0] + ' ' + bi_list[1]
        bigram_cpt_list.append(bigram_cpt)
        multigram_cpt_list.append(bigram_cpt)

    for trigram_idx in range(word_list_len - 2):
        tri_list = word_list[trigram_idx:3 + trigram_idx]
        trigram_cpt = tri_list[0] + ' ' + tri_list[1] + ' ' + tri_list[2]
        trigram_cpt_list.append(trigram_cpt)
        multigram_cpt_list.append(trigram_cpt)

    for fourgram_idx in range(word_list_len - 3):
        four_list = word_list[fourgram_idx:4 + fourgram_idx]
        fourgram_cpt = four_list[0] + ' ' + four_list[1] + ' ' + four_list[2] + ' ' + four_list[3]
        fourgram_cpt_list.append(fourgram_cpt)
        multigram_cpt_list.append(fourgram_cpt)

    multigram_cpt_list.extend(word_list)
    return multigram_cpt_list


def count_concept_(text, concept_dic, concept_count):
    for i, line in enumerate(text):
        for concept in split_text_ngram(line):
            if concept in concept_dic.keys():
                if concept not in concept_count.keys():
                    concept_count[concept] = 1
                else:
                    concept_count[concept] += 1
    return concept_count


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

def add_new_concepts(previous_curri, new_concepts):

    for key in new_concepts.keys():
        previous_curri[key] = 0
    return previous_curri