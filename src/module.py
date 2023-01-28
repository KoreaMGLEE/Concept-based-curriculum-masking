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