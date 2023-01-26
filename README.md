## Efficeint Pre-training of Masked Language Model via Concept-Based Curriculum Masking
Concept-based Curriculum Masking (CCM) is a training strategy for efficient language model pre-training. It can be used for pre-training transformers with relatively lower compute costs. Our framework masks concepts within sentences in easy-to-difficult order. CCM achieves comparative performance with original BERT by only using 1/2 compute costs on the [GLUE benchmark](https://gluebenchmark.com/).

This repository contains code for our EMNLP 2022 paper: [Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking](https://arxiv.org/abs/2212.07617). 

## Results   

| Models        | Cause  | Result | Pairwise|
| :---:         | :---:  | :---:  | :---:   |
| BERT (small)    |  0.0   | 0.0    | 0.0     |
| RoBERTa-large |  52.3  | 32.6   | 19.9    |
| DeBERTa-large |  22.5  | 5.6    | 0.7     |
| GPT-large     |  17.2  | 25.5   | 14.2    |
| GPT-xl        |  35.7  | 38.6   | 3.9     |
| T5-base       |  21.6  | 25.0   | 8.8     |
| T5-large      |  55.5  | 47.2   | 29.5    |
| T5-3b         |  70.7  | 77.3   | 58.0    |
| T5-11b        |  80.1  | 81.8   | 67.2    |
| **Human**     |**97.3**|**97.6**|**97.3** |
