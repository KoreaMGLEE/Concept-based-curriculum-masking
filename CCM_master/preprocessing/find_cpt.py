import os
import time
import pickle
from transformers import *
from multiprocessing import Process
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def split_doc_multigram(doc):
    word_list = doc.split()
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


def find_cpt_c1(datapoint, cpt_dic):
    doc = str()
    doc2 = str()
    datapoint['pure_txt'] = tokenizer.decode(datapoint['encoded_txt'])
    for text in datapoint['pure_txt']:
        doc = doc + text.replace("."," . ").replace(","," , ")
        doc2 = doc2 + text
    cpt1_dic = {}
    multigram_cpt_list = split_doc_multigram(doc)
    for cpt1 in multigram_cpt_list:
        if cpt1 not in cpt1_dic.keys():
            cpt1_dic[cpt1] = 0

    multigram_cpt_list2 = split_doc_multigram(doc2)
    for cpt2 in multigram_cpt_list2:
        if cpt2 not in cpt1_dic.keys():
            multigram_cpt_list.append(cpt2)

    cpt_in_doc = []
    for multigram_cpt in multigram_cpt_list:
        if multigram_cpt in cpt_dic.keys():
            cpt_in_doc.append(multigram_cpt)
    return datapoint, cpt_in_doc, doc


def find_cpt(datapoint, cpt_dic):
    doc = str()
    doc2 = str()
    for text in datapoint['pure_txt']:
        doc = doc + text.replace("."," . ").replace(","," , ")
        doc2 = doc2 + text
    cpt1_dic = {}
    multigram_cpt_list = split_doc_multigram(doc)
    for cpt1 in multigram_cpt_list:
        if cpt1 not in cpt1_dic.keys():
            cpt1_dic[cpt1] = 0

    multigram_cpt_list2 = split_doc_multigram(doc2)
    for cpt2 in multigram_cpt_list2:
        if cpt2 not in cpt1_dic.keys():
            multigram_cpt_list.append(cpt2)

    cpt_in_doc = []
    for multigram_cpt in multigram_cpt_list:
        if multigram_cpt in cpt_dic.keys():
            cpt_in_doc.append(multigram_cpt)
    return cpt_in_doc, doc

load_dir = '../'
save_dir = '../'
curriculum_dir = '../'

def add_find_cpt(file_name, load_dir):
    start_time = time.time()
    with open(os.path.join(load_dir, file_name), 'rb') as f:
        dataset = pickle.load(f)

    print(len(dataset))
    # toy_set = dataset[:1000]
    new_set = []
    for curriculum_num in range(1, 4):
        with open(os.path.join(curriculum_dir, 'curriculum' + str(curriculum_num)), 'rb') as f:
            cpt_dic = pickle.load(f)
        if curriculum_num == 1:
            for datapoint_num, encoded_txt in enumerate(dataset):
                datapoint = {}
                datapoint['encoded_txt'] = encoded_txt
                datapoint, find_cpt_list, doc = find_cpt_c1(datapoint, cpt_dic)
                datapoint['curriculum'+str(curriculum_num)+'_find_cpt_list'] = find_cpt_list
                if datapoint_num % 100000 == 0:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)
                    print("now in %d, %f" %(datapoint_num, time.time() - start_time))
                    start_time = time.time()
                if datapoint_num == 10:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)

                if datapoint_num == 1000:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)
                new_set.append(datapoint)
        else:
            for datapoint_num, datapoint in enumerate(new_set):
                find_cpt_list, doc = find_cpt(datapoint, cpt_dic)
                datapoint['curriculum' + str(curriculum_num) + '_find_cpt_list'] = find_cpt_list
                if datapoint_num % 100000 == 0:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)
                    print("now in %d, %f" % (datapoint_num, time.time() - start_time))
                    start_time = time.time()
                if datapoint_num == 10:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)
                if datapoint_num == 1000:
                    print(len(datapoint['curriculum1_find_cpt_list']), datapoint)
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(new_set, f)

if __name__ == '__main__':
    sub_dir_list = os.listdir(load_dir)
    print(sub_dir_list)

    key_n_tokenzied_doc = {}
    if __name__ == '__main__':
        procs = []
        for iters in range(3):
            max_process_num = 10
            for process_num in range(max_process_num):
                proc = Process(target=add_find_cpt, args=(sub_dir_list[iters*10+process_num], load_dir))
                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()