## Efficeint Pre-training of Masked Language Model via Concept-Based Curriculum Masking
Concept-based Curriculum Masking (CCM) is a training strategy for efficient language model pre-training. It can be used for pre-training transformers with relatively lower compute costs. Our framework masks concepts within sentences in easy-to-difficult order. CCM achieves comparative performance with original BERT by only using 1/2 compute costs on the [GLUE benchmark](https://gluebenchmark.com/).

This repository contains code for our EMNLP 2022 paper: [Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking](https://arxiv.org/abs/2212.07617). For a detailed description and experimental results, please refer to the paper. 

## Results   

Results on the GLUE dev set
| Models               | CoLA | SST  | MRPC | STS  | RTE  |
| :---:                | :---:| :---:| :---:| :---:| :---:|
| BERT (small, 14M)    | 38.0 | 88.7 | 82.8 | 82.0 | 59.2 | 
| CCM (small, 14M)     | 42.8 | 89.1 | 84.1 | 83.3 | 61.3 |
| BERT (medium, 26M)   | 44.9 | 89.6 | 85.4 | 82.7 | 60.3 | 
| CCM (medium, 26M)    | 48.0 | 90.9 | 86.7 | 83.6 | 61.4 | 
| BERT (base, 110M)    | 49.7 | 90.8 | 87.8 | 85.4 | __67.8__ | 
| __CCM (base, 110M)__     | __60.3__ | __93.1__ | __88.3__ | __85.5__ | 65.0  | 

| Models               |  MNLI | QQP  | QNLI | 
| :---:                | :---: | :---:| :---:|
| BERT (small, 14M)    |  76.8 | 88.4 | 85.8 | 
| CCM (small, 14M)     |  77.5 | 88.6 | 86.3 |
| BERT (medium, 26M)   |  78.9 | 89.4 | 87.6 | 
| CCM (medium, 26M)    |  80.0 | 89.2 | 87.6 |
| BERT (base, 110M)    |  81.7 | 90.4 | 89.5 | 
| __CCM (base, 110M)__ |  __84.1__ | __91.0__ | __91.4__ | 

## Requirements 
 - Python 3 
 - Transformers 1.1
 - Numpy 
 - pytorch

## Download conceptnet
Download ConceptNet assertions.   

```
# Download assertions in the data folder.
$ wget ./data/assertions.csv https://s3.amazonaws.com/conceptnet/precomputed-data/2016/assertions/conceptnet-assertions-5.5.0.csv.gz

# run concept_extraction.py 
$ python ./script/concept_extraction.py
```


## Pre-training 
#### Curriculum Construction
Use ```./script/basicconcept_selection.py``` to create the first stage of the curriculum with basic concepts which are connected with many other concepts in the knowledge graph and frequently occur in the pre-training corpus.
 - ```--conceptnet_path```: A path to preprecessed conceptnet file. 
 - ```--topk_connected_concepts```: Top k concepts that are connected to many other concepts in the knowledge graph. 
 - ```--corpus_dir```: A directory containing raw text files to turn into MLM pre-training examples.
 - ```--delete_threshold```: Frequency threshold to filtering rare concepts.
 - ```--basicConcepts_num```: Set the number of basic concepts used for the curriculum. 
 - ```--save_path```: A path to save the set of basic concepts.

Use ```./script/curriculum_construction.py``` to construct concept-based curriculum with basic concepts. 
 - ```--conceptnet_path```: A path to preprecessed conceptnet file. 
 - ```--num_of_hops```: Set the number of hops for adding related concepts to the next stage concept set.
 - ```--basic_concept_path```: A path to load the set of basic concepts.
 - ```--save_dir```: A path to save the set of concepts for each stage of the curriculum.
 - ```--num_of_stages```: Set the number of stage for the curriculum.

#### Data Pre-processing 
Use ```./script/curriculum_construction.py``` to identify concepts in the corpus and arrange them along with the curriculum. 
 - ```--corpus_dir```: A directory containing raw text files to turn into MLM pre-training examples.
 - ```--save_dir```:  A path to save the pre-processed corpus.
 - ```--curriculum_dir```: A directory containing the concept-based curriculum.
 - ```--process_num```: Set the number of CPU processors for the pre-processing.

 
#### Pre-train the model 
Finally, use ```./script/pre-training.py``` to pre-train your models with the concept-based curriculum masking.

 - ```--curriculum_dir```: A directory containing the concept-based curriculum.
 - ```--lr```: Set the learning rate.
 - ```--epochs```: Set the number of epochs. 
 - ```--batch_size```: Set the batch size for conducting at once. 
 - ```--step_batch_size```: Set the batch size for updating per each step (If the memory of GPU is enough, set the batch_size and step_batch_size the same.
 - ```--data_path```: A directory containing pre-processed examples.
 - ```--warmup_steps```: Set the number of steps to warmup the model with the original MLM. 
 - ```--model_size```: Choose the size of the model to pre-train.


## Contact Info 
For help or issues using CCM, please submit a GitHub issue. 

For personal communication related to CCM, please contact Mingyu Lee ```<decon9201@korea.ac.kr>``` or Jun-Hyung Park ```<irish07@korea.ac.kr>```.
