import os
import re
import copy
import pickle
import argparse


def construct_next_stage(previous_concept_set, conceptnet, seen_concepts):
    next_stage_curriculum = {}
    for c in previous_concept_set.keys():
        if c not in seen_concepts.keys():
            seen_concepts[c] = previous_concept_set[c]

            for related_c in conceptnet[c].keys():
                if related_c not in previous_concept_set.keys():
                    next_stage_curriculum[related_c] = len(conceptnet[related_c])
                else:
                    pass
        else:
            next_stage_curriculum[c] = len(conceptnet[c])
            continue
    return next_stage_curriculum, seen_concepts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conceptnet_path", default='../data/preprocessed_conceptnet', type=str)
    parser.add_argument("--num_of_hops", default=1, type=int)
    parser.add_argument("--basic_concept_path", default='../data/basic_concepts', type=str)
    parser.add_argument("--save_dir", default='../data/concept_based_curriculum/', type=str)
    parser.add_argument("--num_of_stages", default=3, type=int)
    args = parser.parse_args()

    with open(args.conceptnet_path, 'rb') as f: conceptnet = pickle.load(f)
    with open(args.basic_concept_path, 'rb') as f: basic_concept_set = pickle.load(f)

    # create_next_stage_curriculum
    curriculum_stage = 1
    with open(os.path.join(args.save_dir, f'curriculum_{curriculum_stage}'), 'wb') as f: pickle.dump(basic_concept_set, f)

    current_conceptset = basic_concept_set
    seen_concepts = {}
    for s in range(args.num_of_stages-2):
        curriculum_stage += 1
        for i in range(args.num_of_hops):
            current_conceptset, seen_concepts = construct_next_stage(current_conceptset, conceptnet, seen_concepts)
        with open(os.path.join(args.save_dir, f'curriculum_{curriculum_stage}'), 'wb') as f: pickle.dump(current_conceptset, f)


    final_stage_conceptset = {}
    for c in conceptnet.keys():
        if c not in seen_concepts.keys():
            final_stage_conceptset[c] = len(conceptnet[c].keys())

    with open(os.path.join(args.save_dir, f'curriculum_{curriculum_stage+1}'), 'wb') as f: pickle.dump(final_stage_conceptset, f)