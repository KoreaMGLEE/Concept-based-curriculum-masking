import os
import copy
import pickle
import argparse
from transformers import BertTokenizer
from multiprocessing import Process
from src.module import split_text_ngram

def find_cpt(datapoint, cpt_dic):
    doc = str()
    doc2 = str()


    for text in datapoint['original_text']:
        doc = doc + text.replace("."," . ").replace(","," , ")
        doc2 = doc2 + text
    cpt1_dic = {}
    multigram_cpt_list = split_text_ngram(doc)
    for cpt1 in multigram_cpt_list:
        if cpt1 not in cpt1_dic.keys():
            cpt1_dic[cpt1] = 0

    multigram_cpt_list2 = split_text_ngram(doc2)
    for cpt2 in multigram_cpt_list2:
        if cpt2 not in cpt1_dic.keys():
            multigram_cpt_list.append(cpt2)

    cpt_in_doc = []
    for multigram_cpt in multigram_cpt_list:
        if multigram_cpt in cpt_dic.keys():
            cpt_in_doc.append(multigram_cpt)
    return cpt_in_doc, doc


def add_find_cpt(file_name, load_dir):
    with open(os.path.join(load_dir, file_name), 'r') as f:
        lines = f.readlines()
    new_set = []
    dataset = {}
    datapoint = {}

    encoded_text = []
    original_str = str()
    print(len(lines))
    data_id = 0
    num_add_ids = 0
    for line in lines[:1000]:
        ## tokenizer.batch_encode is much faster than tokenizer.encode
        ## If you want to use a huge pre-training corpus, just modify the code here a bit.
        add_ids = tokenizer.encode(line)[1:-1]
        if len(encoded_text) + len(add_ids) > 510:
            datapoint['encoded_txt'] = encoded_text
            datapoint['original_text'] = original_str
            dataset[f'id_{data_id}'] = {'encoded_txt': encoded_text,
                                  'original_text': original_str}
            encoded_text = add_ids
            original_str = line
            data_id += 1
            num_add_ids = 0
        else:
            num_add_ids += 1
            encoded_text.extend(add_ids)
            original_str = original_str + ' ' + line


    for curriculum_num in range(len(os.listdir(args.curriculum_dir))):
        with open(os.path.join(args.curriculum_dir, 'curriculum_' + str(curriculum_num+1)), 'rb') as f:
            curriulum_concept_set = pickle.load(f)

        if curriculum_num == 0:
            for d_ in dataset.keys():
                find_cpt_list, doc = find_cpt(dataset[d_], curriulum_concept_set)
                dataset[d_][f'curriculum_{curriculum_num+1}_concepts'] = set(find_cpt_list)
                new_d = copy.deepcopy(dataset[d_])
                new_set.append(new_d)

        else:
            with open(os.path.join(args.curriculum_dir, 'curriculum_' + str(curriculum_num)), 'rb') as f:
                previous_concept_set = pickle.load(f)

            for concept_ in previous_concept_set.keys():
                if concept_ in curriulum_concept_set.keys():
                    del curriulum_concept_set[concept_]

            for d_point in new_set:
                find_cpt_list, doc = find_cpt(d_point, curriulum_concept_set)
                d_point[f'curriculum_{curriculum_num+1}_concepts'] = find_cpt_list

    with open(os.path.join(args.save_dir, file_name.split('.')[0]), 'wb') as f:
        pickle.dump(new_set, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default='../data/pre-training-corpus/', type=str)
    parser.add_argument("--save_dir", default='../data/preprocessed_corpus/', type=str)
    parser.add_argument("--curriculum_dir", default='../data/concept_based_curriculum/', type=str)
    parser.add_argument("--process_num", default=2, type=int)
    args = parser.parse_args()

    save_dir = args.save_dir
    curriculum_dir = args.curriculum_dir

    sub_dir_list = os.listdir(args.corpus_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    key_n_tokenzied_doc = {}

    if __name__ == '__main__':
        procs = []
        max_process_num = args.process_num
        for process_num in range(max_process_num):
            proc = Process(target=add_find_cpt, args=(sub_dir_list[process_num], args.corpus_dir))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()