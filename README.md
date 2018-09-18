# Pseudo-references Generation

This project is the implementation of pseudo-reference generation algorithm proposed in EMNLP 2018 paper: [Multi-Reference Training with Pseudo-References for Neural Translation and Text Generation](https://arxiv.org/abs/1808.09564).

## Dependencies

- python: 2.7
- pytrch: 0.3.1
- torchtext: 0.2.1
- networkx: 2.0
- numpy: 1.13.3
- sklearn: 0.19.1
- matplotlib: 2.1.0
- scipy: 0.19.1
- nltk: 3.2.4

## Lattice Generation

This project includes both hard word alignment and soft word alignment algorithms to generate lattice. You can use the [coco-caption](!http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) dataset or a small dataset extracted from coco we provide at 'data/dataset_small.json'. You can generate lattice with hard or soft word alignment algorithm via the following example commands.

```
python lattice.py -order_method hard -align_method hard -dataset data/dataset_small.json -minus 0.5
python lattice.py -order_method soft -align_method soft -dataset data/dataset_small.json -minus 0.6 -lm_dictionary data/LM_coco.dict -lm_model data/LM_coco.pth
```

- align_method [soft|hard]
    - Soft or hard alignment described in the paper
- order_method [soft|hard|random]
    - Sort original sentences before merging.
- minus
    - Global penalty $p$ described in the paper to avoid merge unrelated words.
- gpuid
    - Set GPU for soft word alignment
- save_graph
    - If True, the lattice graph will be saved instead of pseudo-references
- multi_process
    - Enable multi-processing to speedup generation
- n_cpu
    - Number of threads will be used in multi-processing 
- dataset
    - We provide a small dataset 'data/dataset_small.json' extracted from [coco-caption](!http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). In this dataset there are only three examples including the first one from dev-set which will be involved in pseudo-ref generation algorithm.
- lm_model
    - The language model used in soft word alignment algorithm
- lm_dictionary
    - The dictionary file of the language model used in soft word alignment algorithm

The output will be a json file stalled as 'data/dataset\_(ORDER_METHOD)\_(ALIGNMENT_METHOD)_(MINUS)'.

## Bidirectional Language Model

A bidirectional language model will be used in the soft sentence alignment algorithm. Our implementation is included in the folder 'language_model'. We provide a model trained on MSCOCO at 'data/LM_coco.pth' and it's corresponding dictionary data 'data/LM_coco.dict'. (Please note that this language model is slightly different to the one used in paper, so the output lattice maybe different.)


## Lattice Visualization

We provide codes to visualize generated lattice by converting it into LaTex. For example, you can use the following command to print the 1st (start from 0) lattice in json file data/dataset_soft_soft_0.60.json which is generated from data/dataset_small.json.

```
python lattice2latex.py -original_dataset data/dataset_small.json -lattice_dataset data/dataset_soft_soft_0.60.json -lattice_index 1
```

