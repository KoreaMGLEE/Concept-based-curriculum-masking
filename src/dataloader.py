from torch.utils.data import Dataset
import pickle
from transformers import *
import random
import copy
import torch
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class create_dataset_base_dynamic_ELECTRA(Dataset):
    def __init__(self, load_dir, file_name, curriculum_num, tokenizer):
        print("create_dataset...")
        with open(os.path.join(load_dir, file_name), 'rb') as f:
            self.datas = pickle.load(f)

        self.tokenizer = tokenizer
        self.curriculum_num = curriculum_num

        print("now we are in %d curriculum " % int(self.curriculum_num))

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def split_doc_multigram(self, doc):
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

    def find_cpt(self, datapoint):
        doc = str()
        for text in datapoint['pure_txt']:
            doc = doc + text

        multigram_cpt_list = self.split_doc_multigram(doc)
        cpt_in_doc = []

        for multigram_cpt in multigram_cpt_list:
            if multigram_cpt in self.cpt_count.keys():
                cpt_in_doc.append(multigram_cpt)

        return cpt_in_doc, doc

    def encoded_cpt_(self, cpt):
        encoded_cpt_list = []
        encoded_cpt_list.append(self.tokenizer(cpt)['input_ids'][1:-1])
        return encoded_cpt_list

    def random_masking_(self, cpt_masked_sentence, lm_position, adjusting_mask_prob):

        masked_sentence = []
        label_mask_ = []

        for lm_id, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:

                if random.random() <= adjusting_mask_prob:
                    lm_position.append(lm_id + 1)
                    label_mask_.append(False)  # masking 할거면 false
                    masked_sentence.append(self.tokenizer.mask_token_id)

                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)

            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def random_masking(self, cpt_masked_sentence, label_mask_, lm_position, adjusting_mask_prob):

        masked_sentence = []

        for position, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:

                if random.random() <= adjusting_mask_prob:
                    lm_position.append(position + 1)
                    label_mask_[position] = False  # masking 할거면 false
                    masked_sentence.append(self.tokenizer.mask_token_id)

                else:
                    masked_sentence.append(id)

            else:
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def spaced_cpt_mask(self, encoded_sentence, encoded_key, mask_count, cpt_mask_tensor, lm_position, mask_prob):
        cursor = 0
        # encoded cpt 위치 찾기
        start_n_end_point_list = []
        for i, encoded_id in enumerate(encoded_sentence):
            if encoded_id == encoded_key[cursor]:
                cursor += 1
                if cursor == len(encoded_key):
                    start_point = i + 1 - cursor
                    end_point = i
                    start_n_end_point_list.append([start_point, end_point])
                    cursor = 0
            else:
                cursor = 0
        # 찾은 위치에 맞춰 masking
        cpt_masked_sen = encoded_sentence
        for start_point, end_point in start_n_end_point_list:
            if random.random() <= mask_prob:
                j = start_point
                for i in range(end_point - start_point + 1):
                    lm_position.append(j + 1)
                    cpt_mask_tensor[j] = False
                    cpt_masked_sen[j] = self.tokenizer.mask_token_id
                    mask_count += 1
                    j += 1

        return cpt_masked_sen, mask_count, cpt_mask_tensor, lm_position

    def __getitem__(self, idx):
        datapoint = self.datas[idx]
        lm_position = []
        # 학습할 corpus에 있는 concept 찾고
        if self.curriculum_num == 0:
            datapoint = self.datas[idx]
            masked_sentence, label_mask_, lm_position = self.random_masking_(datapoint['encoded_txt'], lm_position,
                                                                             0.15)
            datapoint['masking_txt'] = masked_sentence
            datapoint['label_mask'] = label_mask_  # .tolist()
            datapoint['lm_position'] = lm_position
            return datapoint

        elif self.curriculum_num != 4:
            find_cpt_list = []
            for i in range(1, self.curriculum_num + 1):
                find_cpt_list.extend(datapoint['curriculum' + str(i) + '_find_cpt_list'])

            if len(find_cpt_list) != 0:
                count_cpttoken_in_example = 0
                encoded_find_cpt_list = self.tokenizer.batch_encode_plus(find_cpt_list)['input_ids']
                for encoded_cpt_in_list in encoded_find_cpt_list:
                    count_cpttoken_in_example += int(len(encoded_cpt_in_list) - 2)
                mask_prob = float(
                    int(len(datapoint[
                                'encoded_txt']) * 0.15) / count_cpttoken_in_example)  # 마스킹 확률 * 전체 찾은 cpt 수 = 전체 토큰 수 *0.15

            else:
                mask_prob = 0

            find_cpt_list = list(set(find_cpt_list))
            mask_count = 0
            cpt_masked_sendtence = copy.deepcopy(datapoint['encoded_txt'])
            cpt_mask_tensor = [True] * len(datapoint['encoded_txt'])

            random.shuffle(find_cpt_list)
            if find_cpt_list != []:
                encoded_cpt_list = self.tokenizer(find_cpt_list)['input_ids']
                for encoded_cpt in encoded_cpt_list:
                    if mask_count > 19:
                        continue
                    cpt_masked_sendtence, mask_count, cpt_mask_tensor, lm_position = self.spaced_cpt_mask(
                        cpt_masked_sendtence,
                        encoded_cpt, mask_count,
                        cpt_mask_tensor, lm_position, mask_prob)
            # 새로운 masking prob 정의
            datapoint['masking_txt'] = cpt_masked_sendtence
            datapoint['label_mask'] = cpt_mask_tensor
            datapoint['lm_position'] = lm_position
            # for i in range(1, self.curriculum_num + 1):
            #     del datapoint['curriculum' + str(i)+'_find_cpt_list']
            return datapoint

        else:
            find_cpt_list_3 = []
            for i in range(1, self.curriculum_num):
                find_cpt_list_3.extend(datapoint['curriculum' + str(i) + '_find_cpt_list'])

            # masking 할지말지 선택
            count_cpttoken_in_example = 0
            mask_count = 0

            cpt_masked_sendtence = copy.deepcopy(datapoint['encoded_txt'])
            cpt_mask_tensor = [True] * len(datapoint['encoded_txt'])

            if len(find_cpt_list_3) != 0:
                encoded_find_cpt_list = self.tokenizer.batch_encode_plus(find_cpt_list_3)['input_ids']
                for encoded_cpt_in_list in encoded_find_cpt_list:
                    count_cpttoken_in_example += int(len(encoded_cpt_in_list) - 2)
                mask_prob = float(
                    int(len(
                        cpt_masked_sendtence) * 0.10) / count_cpttoken_in_example)  # 마스킹 확률 * 전체 찾은 cpt 수 = 전체 토큰 수 *0.15
            else:
                mask_prob = 0

            find_cpt_list = list(set(find_cpt_list_3))
            random.shuffle(find_cpt_list)

            if find_cpt_list != []:
                encoded_cpt_list = self.tokenizer(find_cpt_list)['input_ids']
                for encoded_cpt in encoded_cpt_list:
                    cpt_masked_sendtence, mask_count, cpt_mask_tensor, lm_position = self.spaced_cpt_mask(
                        cpt_masked_sendtence,
                        encoded_cpt,
                        mask_count,
                        cpt_mask_tensor, lm_position, mask_prob)

            cpt_masked_sendtence, cpt_mask_tensor, lm_position = self.random_masking(cpt_masked_sendtence,
                                                                                     cpt_mask_tensor, lm_position, 0.05)

            # 새로운 masking prob 정의
            datapoint['masking_txt'] = cpt_masked_sendtence
            datapoint['label_mask'] = cpt_mask_tensor
            datapoint['lm_position'] = lm_position
            return datapoint


class create_dataset_base_dynamic(Dataset):
    def __init__(self, load_dir, file_name, curriculum_num, tokenizer):
        print("create_dataset...")
        with open(os.path.join(load_dir, file_name), 'rb') as f:
            self.datas = pickle.load(f)
        #
        # with open('/home/user3/data/pure_huggingface/CCL_clean_corpus_final_v220325_1', 'rb') as f:
        #     self.datas = pickle.load(f)
        self.tokenizer = tokenizer
        self.curriculum_num = curriculum_num
        print("now we are in %d curriculum " % int(self.curriculum_num))

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def split_doc_multigram(self, doc):
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

    def find_cpt(self, datapoint):
        doc = str()
        for text in datapoint['pure_txt']:
            doc = doc + text

        multigram_cpt_list = self.split_doc_multigram(doc)
        cpt_in_doc = []

        for multigram_cpt in multigram_cpt_list:
            if multigram_cpt in self.cpt_count.keys():
                cpt_in_doc.append(multigram_cpt)

        return cpt_in_doc, doc

    def encoded_cpt_(self, cpt):
        encoded_cpt_list = []
        encoded_cpt_list.append(self.tokenizer(cpt)['input_ids'][1:-1])
        return encoded_cpt_list

    def random_masking_(self, cpt_masked_sentence, lm_position, adjusting_mask_prob):

        masked_sentence = []
        label_mask_ = []

        for lm_id, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(lm_id + 1)
                    label_mask_.append(False)  # masking 할거면 false

                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def random_masking(self, cpt_masked_sentence, label_mask_, lm_position, adjusting_mask_prob, mask_count):

        masked_sentence = []

        for position, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if mask_count > 18:
                    masked_sentence.append(id)
                    continue
                else:
                    if random.random() <= adjusting_mask_prob:
                        lm_position.append(position + 1)
                        label_mask_[position] = False  # masking 할거면 false
                        masked_sentence.append(self.tokenizer.mask_token_id)
                        mask_count += 1

                    else:
                        masked_sentence.append(id)

            else:
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def spaced_cpt_mask(self, encoded_sentence, encoded_key, mask_count, cpt_mask_tensor, lm_position, mask_prob):
        cursor = 0
        # encoded cpt 위치 찾기
        start_n_end_point_list = []
        for i, encoded_id in enumerate(encoded_sentence):
            if encoded_id == encoded_key[cursor]:
                cursor += 1
                if cursor == len(encoded_key):
                    start_point = i + 1 - cursor
                    end_point = i
                    start_n_end_point_list.append([start_point, end_point])
                    cursor = 0
            else:
                cursor = 0
        # 찾은 위치에 맞춰 masking
        cpt_masked_sen = encoded_sentence
        for start_point, end_point in start_n_end_point_list:
            if random.random() <= mask_prob:
                if random.random() >= 0.2:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        lm_position.append(j + 1)
                        cpt_mask_tensor[j] = False
                        cpt_masked_sen[j] = self.tokenizer.mask_token_id
                        mask_count += 1
                        j += 1
                elif random.random() <= 0.5:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        lm_position.append(j + 1)
                        cpt_mask_tensor[j] = False
                        cpt_masked_sen[j] = random.randint(1, 30251)
                        mask_count += 1
                        j += 1
                else:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        cpt_mask_tensor[j] = False
                        mask_count += 1
                        j += 1
        return cpt_masked_sen, mask_count, cpt_mask_tensor, lm_position

    def __getitem__(self, idx):

        datapoint = self.datas[idx]
        lm_position = []
        # 학습할 corpus에 있는 concept 찾고
        if self.curriculum_num == 0:
            datapoint = self.datas[idx]
            masked_sentence, label_mask_, lm_position = self.random_masking_(datapoint['encoded_txt'][1:-1],
                                                                             lm_position, 0.15)
            datapoint['masking_txt'] = masked_sentence
            datapoint['label_mask'] = label_mask_  # .tolist()
            datapoint['lm_position'] = lm_position
            total_count = len(label_mask_) - sum(label_mask_)
            datapoint["mask_count_concept"] = 0
            datapoint["total_count"] = total_count
            return datapoint

        else:
            find_cpt_list = []
            mask_prob = 0.18  # + 0.01 * (self.curriculum_num -1)

            for i in range(1, self.curriculum_num + 1):
                find_cpt_list.extend(datapoint['curriculum' + str(i) + '_find_cpt_list'])

            if len(find_cpt_list) != 0:
                count_cpttoken_in_example = 0
                encoded_find_cpt_list = self.tokenizer.batch_encode_plus(find_cpt_list)['input_ids']
                for encoded_cpt_in_list in encoded_find_cpt_list:
                    count_cpttoken_in_example += int(len(encoded_cpt_in_list) - 2)
                mask_prob = float(
                    int(len(datapoint['encoded_txt'][
                            1:-1]) * mask_prob) / count_cpttoken_in_example)  # 마스킹 확률 * 전체 찾은 cpt 수 = 전체 토큰 수 *0.15
            else:
                mask_prob = 0

            find_cpt_list = list(set(find_cpt_list))
            mask_count = 0
            cpt_masked_sendtence = copy.deepcopy(datapoint['encoded_txt'][1:-1])
            cpt_mask_tensor = [True] * len(datapoint['encoded_txt'][1:-1])
            random.shuffle(find_cpt_list)

            if random.random() >= 0.2:
                mask_count_ = 19
            else:
                mask_count_ = 20
            if find_cpt_list != []:
                encoded_cpt_list = self.tokenizer(find_cpt_list)['input_ids']

                for encoded_cpt in encoded_cpt_list:
                    if mask_count >= mask_count_:
                        continue
                    cpt_masked_sendtence, mask_count, cpt_mask_tensor, lm_position = self.spaced_cpt_mask(
                        cpt_masked_sendtence,
                        encoded_cpt[1:-1], mask_count,
                        cpt_mask_tensor, lm_position, mask_prob)

            total_count = len(cpt_mask_tensor) - sum(cpt_mask_tensor)

            # 새로운 masking prob 정의
            datapoint['masking_txt'] = cpt_masked_sendtence
            datapoint['label_mask'] = cpt_mask_tensor
            datapoint['lm_position'] = lm_position

            datapoint["mask_count_concept"] = mask_count
            datapoint["total_count"] = total_count
            return datapoint


class mask_single_concept(Dataset):
    def __init__(self, load_dir, file_name, curriculum_num, tokenizer):
        print("create_dataset...")
        with open(os.path.join(load_dir, file_name), 'rb') as f:
            self.datas = pickle.load(f)

        self.tokenizer = tokenizer
        self.curriculum_num = curriculum_num
        print("now we are in %d curriculum " % int(self.curriculum_num))

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def split_doc_multigram(self, doc):
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

    def find_cpt(self, datapoint):
        doc = str()
        for text in datapoint['pure_txt']:
            doc = doc + text

        multigram_cpt_list = self.split_doc_multigram(doc)
        cpt_in_doc = []

        for multigram_cpt in multigram_cpt_list:
            if multigram_cpt in self.cpt_count.keys():
                cpt_in_doc.append(multigram_cpt)

        return cpt_in_doc, doc

    def encoded_cpt_(self, cpt):
        encoded_cpt_list = []
        encoded_cpt_list.append(self.tokenizer(cpt)['input_ids'][1:-1])
        return encoded_cpt_list

    def spaced_cpt_mask(self, encoded_sentence, encoded_key, mask_count, cpt_mask_tensor, lm_position, mask_prob):
        cursor = 0
        # encoded cpt 위치 찾기
        start_n_end_point_list = []
        for i, encoded_id in enumerate(encoded_sentence):
            if encoded_id == encoded_key[cursor]:
                cursor += 1
                if cursor == len(encoded_key):
                    start_point = i + 1 - cursor
                    end_point = i
                    start_n_end_point_list.append([start_point, end_point])
                    cursor = 0
            else:
                cursor = 0
        # 찾은 위치에 맞춰 masking
        cpt_masked_sen = encoded_sentence
        for start_point, end_point in start_n_end_point_list:
            if random.random() <= mask_prob:
                if random.random() >= 0.2:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        lm_position.append(j + 1)
                        cpt_mask_tensor[j] = False
                        cpt_masked_sen[j] = self.tokenizer.mask_token_id
                        mask_count += 1
                        j += 1
                elif random.random() <= 0.5:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        lm_position.append(j + 1)
                        cpt_mask_tensor[j] = False
                        cpt_masked_sen[j] = random.randint(1, 30251)
                        mask_count += 1
                        j += 1
                else:
                    j = start_point
                    for i in range(end_point - start_point + 1):
                        cpt_mask_tensor[j] = False
                        mask_count += 1
                        j += 1
        return cpt_masked_sen, mask_count, cpt_mask_tensor, lm_position

    def __getitem__(self, idx):

        datapoint = self.datas[idx]
        lm_position = []

        find_cpt_list = []
        mask_prob = 0.18  # + 0.01 * (self.curriculum_num -1)

        for i in range(1, self.curriculum_num + 1):
            find_cpt_list.extend(datapoint['curriculum' + str(i) + '_find_cpt_list'])

        if len(find_cpt_list) != 0:
            count_cpttoken_in_example = 0
            encoded_find_cpt_list = self.tokenizer.batch_encode_plus(find_cpt_list)['input_ids']
            for encoded_cpt_in_list in encoded_find_cpt_list:
                count_cpttoken_in_example += int(len(encoded_cpt_in_list) - 2)
            mask_prob = float(
                int(len(datapoint['encoded_txt'][
                        1:-1]) * mask_prob) / count_cpttoken_in_example)  # 마스킹 확률 * 전체 찾은 cpt 수 = 전체 토큰 수 *0.15
        else:
            mask_prob = 0

        find_cpt_list = list(set(find_cpt_list))
        mask_count = 0
        cpt_masked_sendtence = copy.deepcopy(datapoint['encoded_txt'][1:-1])
        cpt_mask_tensor = [True] * len(datapoint['encoded_txt'][1:-1])
        random.shuffle(find_cpt_list)

        if random.random() >= 0.2:
            mask_count_ = 19
        else:
            mask_count_ = 20
        if find_cpt_list != []:
            encoded_cpt_list = self.tokenizer(find_cpt_list)['input_ids']

            for encoded_cpt in encoded_cpt_list:
                if mask_count >= mask_count_:
                    continue
                cpt_masked_sendtence, mask_count, cpt_mask_tensor, lm_position = self.spaced_cpt_mask(
                    cpt_masked_sendtence,
                    encoded_cpt[1:-1], mask_count,
                    cpt_mask_tensor, lm_position, mask_prob)

        total_count = len(cpt_mask_tensor) - sum(cpt_mask_tensor)

        # 새로운 masking prob 정의
        datapoint['masking_txt'] = cpt_masked_sendtence
        datapoint['label_mask'] = cpt_mask_tensor
        datapoint['lm_position'] = lm_position

        datapoint["mask_count_concept"] = mask_count
        datapoint["total_count"] = total_count
        return datapoint


class create_dataset_bert_length(Dataset):
    def __init__(self, load_dir, file_name, curriculum_num, tokenizer):
        print("create_dataset...")
        with open(os.path.join(load_dir, file_name), 'rb') as f:
            self.datas = pickle.load(f)
        self.tokenizer = tokenizer
        self.curriculum_num = curriculum_num
        print("now we are in %d curriculum " % int(self.curriculum_num))

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def random_masking_(self, cpt_masked_sentence, lm_position, adjusting_mask_prob):

        masked_sentence = []
        label_mask_ = []

        for lm_id, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(lm_id + 1)
                    label_mask_.append(False)  # masking 할거면 false

                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def __getitem__(self, idx):
        lm_position = []
        # 학습할 corpus에 있는 concept 찾고
        datapoint = self.datas[idx]
        encoded_data = datapoint['encoded_txt'][1:-1]
        end = 64 * (self.curriculum_num + 1) - 2
        masked_sentence, label_mask_, lm_position = self.random_masking_(encoded_data[:end],
                                                                         lm_position, 0.15)

        datapoint['masking_txt'] = masked_sentence
        datapoint['label_mask'] = label_mask_  # .tolist()
        datapoint['lm_position'] = lm_position
        total_count = len(label_mask_) - sum(label_mask_)
        datapoint["mask_count_concept"] = 0
        datapoint["total_count"] = total_count
        return datapoint


class create_dataset_base_dynamic_gradual_masking(Dataset):
    def __init__(self, load_dir, file_name, curriculum_num, tokenizer):
        print("create_dataset...")
        with open(os.path.join(load_dir, file_name), 'rb') as f:
            self.datas = pickle.load(f)
        self.tokenizer = tokenizer
        self.curriculum_num = curriculum_num
        print("now we are in %d curriculum " % int(self.curriculum_num))

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def random_masking_(self, cpt_masked_sentence, lm_position, adjusting_mask_prob):

        masked_sentence = []
        label_mask_ = []

        for lm_id, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(lm_id + 1)
                    label_mask_.append(False)  # masking 할거면 false

                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, label_mask_, lm_position

    def __getitem__(self, idx):

        lm_position = []
        # 학습할 corpus에 있는 concept 찾고
        datapoint = self.datas[idx]
        masking_prob = 0.15 * (1 - 1 / 3 * (
                    4 - self.curriculum_num) / 4)  # 마지막에 0.15 * 1 처음에 0.15 * 1/3 이니까 (0.15 * 1- 0.15 * 2/3 * (4-self.curriculum_num)/4 ) )
        masked_sentence, label_mask_, lm_position = self.random_masking_(datapoint['encoded_txt'][1:-1],
                                                                         lm_position, masking_prob)
        datapoint['masking_txt'] = masked_sentence
        datapoint['label_mask'] = label_mask_  # .tolist()
        datapoint['lm_position'] = lm_position
        total_count = len(label_mask_) - sum(label_mask_)
        datapoint["mask_count_concept"] = 0
        datapoint["total_count"] = total_count
        return datapoint


class create_dataset_Electra(Dataset):
    def __init__(self, mode, tokenizer):
        print("create_dataset..." + mode)
        with open('/home/user10/origin/' + mode, 'rb') as f:
            self.datas = pickle.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)

    def random_masking(self, cpt_masked_sentence, adjusting_mask_prob, mask_count):
        masked_sentence = []
        label_mask_ = []
        lm_position = []
        for id_position, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(id_position + 1)
                    mask_count += 1
                    label_mask_.append(False)  # masking 할거면 false
                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, torch.BoolTensor(label_mask_), lm_position, mask_count

    def __getitem__(self, idx):
        # datapoint = self.datas[idx]
        # 학습할 corpus에 있는 concept 찾고
        datapoint = self.datas[idx]
        mask_count = 0
        masked_sentence, label_mask_, lm_position, mask_count = self.random_masking(datapoint['encoded_txt'], 0.15,
                                                                                    mask_count)
        datapoint['masking_txt'] = masked_sentence
        datapoint['label_mask'] = label_mask_.tolist()
        datapoint['lm_position'] = lm_position
        return datapoint


def padded_sequence(samples):
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []

    concept_mask_count = []
    total_mask_count = []

    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        LM_label.append(sample['encoded_txt'][1:-1])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])

        concept_mask_count.append(sample["mask_count_concept"])
        total_mask_count.append(sample["total_count"])

        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max >= 128:
        LM_max = 128
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_example)))
            lm_label_batch.append(
                [tokenizer.cls_token_id] + LM_label[i] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_label[i])))
            label_mask_batch.append([True] + label_mask_[i] + [True] + [True] * (LM_max - len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:128] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id] + LM_label[i][:128] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:128] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position, concept_mask_count, total_mask_count


def padded_sequence_bert_length_64(samples):
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []

    concept_mask_count = []
    total_mask_count = []

    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        encoded_data = sample['encoded_txt'][1:-1]
        LM_label.append(encoded_data[:len(sample['masking_txt'])])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])

        concept_mask_count.append(sample["mask_count_concept"])
        total_mask_count.append(sample["total_count"])

        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []

    if LM_max >= 64:
        LM_max = 64

    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_example)))
            lm_label_batch.append(
                [tokenizer.cls_token_id] + LM_label[i] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_label[i])))
            label_mask_batch.append([True] + label_mask_[i] + [True] + [True] * (LM_max - len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:LM_max] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id] + LM_label[i][:LM_max] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:LM_max] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position, concept_mask_count, total_mask_count


def padded_sequence_bert_length_128(samples):
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []

    concept_mask_count = []
    total_mask_count = []

    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        encoded_data = sample['encoded_txt'][1:-1]
        LM_label.append(encoded_data[:len(sample['masking_txt'])])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])

        concept_mask_count.append(sample["mask_count_concept"])
        total_mask_count.append(sample["total_count"])

        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max >= 128:
        LM_max = 128
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_example)))
            lm_label_batch.append(
                [tokenizer.cls_token_id] + LM_label[i] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_label[i])))
            label_mask_batch.append([True] + label_mask_[i] + [True] + [True] * (LM_max - len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:128] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id] + LM_label[i][:128] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:128] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position, concept_mask_count, total_mask_count


def padded_sequence_bert_length_256(samples):
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []

    concept_mask_count = []
    total_mask_count = []

    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        encoded_data = sample['encoded_txt'][1:-1]
        LM_label.append(encoded_data[:len(sample['masking_txt'])])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])

        concept_mask_count.append(sample["mask_count_concept"])
        total_mask_count.append(sample["total_count"])

        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max >= 256:
        LM_max = 256
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_example)))
            lm_label_batch.append(
                [tokenizer.cls_token_id] + LM_label[i] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_label[i])))
            label_mask_batch.append([True] + label_mask_[i] + [True] + [True] * (LM_max - len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:128] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id] + LM_label[i][:128] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:128] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position, concept_mask_count, total_mask_count


def padded_sequence_bert_length_512(samples):
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []

    concept_mask_count = []
    total_mask_count = []

    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        encoded_data = sample['encoded_txt'][1:-1]
        LM_label.append(encoded_data[:len(sample['masking_txt'])])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])

        concept_mask_count.append(sample["mask_count_concept"])
        total_mask_count.append(sample["total_count"])

        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max >= 512:
        LM_max = 512
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_example)))
            lm_label_batch.append(
                [tokenizer.cls_token_id] + LM_label[i] + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                            LM_max - len(LM_label[i])))
            label_mask_batch.append([True] + label_mask_[i] + [True] + [True] * (LM_max - len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:128] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id] + LM_label[i][:128] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:128] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position, concept_mask_count, total_mask_count

