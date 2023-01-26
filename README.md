## Efficeint Pre-training of Masked Language Model via Concept-Based Curriculum Masking
Concept-based Curriculum Masking (CCM) is a training strategy for efficient language model pre-training. It can be used for pre-training transformers with relatively lower compute costs. Our framework masks concepts within sentences in easy-to-difficult order. CCM achieves comparative performance with original BERT by only using 1/2 compute costs on the [GLUE benchmark](https://gluebenchmark.com/).

This repository contains code for our EMNLP 2022 paper: [Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking](https://arxiv.org/abs/2212.07617). 

## Results   

Results on the GLUE dev set
| Models               | CoLA | SST  | MRPC | STS  | RTE  | MNLI | QQP  | QNLI | Avg. |
| :---:                | :---:| :---:| :---:| :---:| :---:|:---: | :---:| :---:| :---:|
| BERT (small, 14M)    | 38.0 | 88.7 | 82.8 | 82.0 | 59.2 | 76.8 | 88.4 | 85.8 | 75.2 |
| CCM (small, 14M)     | 42.8 | 89.1 | 84.1 | 83.3 | 61.3 | 77.5 | 88.6 | 86.3 | 76.6 |
| BERT (medium, 26M)   | 44.9 | 89.6 | 85.4 | 82.7 | 60.3 | 78.9 | 89.4 | 87.6 | 77.4 |
| CCM (medium, 26M)    | 48.0 | 90.9 | 86.7 | 83.6 | 61.4 | 80.0 | 89.2 | 87.6 | 78.4 |
| BERT (base, 110M)    | 49.7 | 90.8 | 87.8 | 85.4 | __67.8__ | 81.7 | 90.4 | 89.5 | 80.4 |
| __CCM (base, 110M)__     | __60.3__ | __93.1__ | __88.3__ | __85.5__ | 65.0  | __84.1__ | __91.0__ | __91.4__ | __82.3__ |

## Pre-training 
