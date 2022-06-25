import os
import re
import time
import pickle
from transformers import *
from multiprocessing import Process
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def split_doc_multigram(doc):
    word_list = doc.split()
    res = list(filter(lambda ele: re.search("[a-zA-Z0-9\s]+", ele) is not None, word_list))

    return res

def find_cpt(datapoint, cpt_dic):
    doc = str()

    for text in datapoint['pure_txt']:
        doc = doc + text.replace(".","").replace(",","")

    cpt_list = split_doc_multigram(doc)
    cpt_in_doc = []
    cpt_dic['[SEP]'] = 0
    cpt_dic['[CLS]'] = 0

    for multigram_cpt in cpt_list:
        if multigram_cpt not in cpt_dic.keys():
            cpt_in_doc.append(multigram_cpt)

    return cpt_in_doc, doc

def add_mask_prob(file_name, load_dir, save_dir):
    start_time = time.time()
    with open(os.path.join(load_dir, file_name), 'rb') as f:
        dataset = pickle.load(f)
    full_cpt_dic = {}
    for i in range(1, 4):
        with open('/home/user3/Third_paper/data/curriculum_hop/basic_curriculum' + str(i), 'rb') as f:
            tmp_cpt_dic = pickle.load(f)
        for key in tmp_cpt_dic.keys():
            full_cpt_dic[key] = 0

    for datapoint_num, datapoint in enumerate(dataset):
        find_cpt_list, doc = find_cpt(datapoint, full_cpt_dic)
        datapoint['curriculum' + str(4) + '_find_cpt_list'] = find_cpt_list

        if datapoint_num % 1000000 == 0:
            print(len(datapoint['curriculum1_find_cpt_list']))
            print("datapoint_num : %d %f" %(datapoint_num, time.time()-start_time))
            print(datapoint)
            start_time = time.time()
        if 'pure_txt' in datapoint.keys():
            del datapoint['pure_txt']

    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(dataset, f)

file_name = 1
load_dir = '/home/user3/data/pure_huggingface/512_length/CCL_clean_corpus_512/'
save_dir = '/home/user3/data/pure_huggingface/512_length/CCL_clean_corpus_final_512/'
if __name__ == '__main__':
    sub_dir_list = os.listdir(load_dir)
    print(sub_dir_list)
    procs = []
    for iter in range(3):
        max_process_num = 10
        for process_num in range(max_process_num):
            proc = Process(target=add_mask_prob, args=(sub_dir_list[iter*10+process_num], load_dir, save_dir))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()