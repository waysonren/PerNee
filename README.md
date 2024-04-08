# Nested Event Extraction upon Pivot Element Recognition 

The ACE2005-Nest dataset and the code of the PerNee model for COLING 2024 paper: [Nested Event Extraction upon Pivot Element Recognition](https://arxiv.org/abs/2309.12960)


## 1. The ACE2005-Nest dataset
Due to licensing limitations, we can only provide the nested event annotations. You may download the ACE2005 dataset from the [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) and merge it with our nested annotation data.

### (1) ACE2005 download and preprocess

The ACE2005 dataset can be download in [LDC](https://catalog.ldc.upenn.edu/LDC2006T06).

We the preprocess the ACE2005 dataset following [OneIE](https://blender.cs.illinois.edu/software/oneie/).

`python preprocessing/process_ace.py -i <INPUT_DIR>/LDC2006T06/data -o <OUTPUT_DIR>
  -s resource/splits/ACE05-E -b bert-large-cased -c <BERT_CACHE_DIR> -l english`
  
The format of preprocessed data is as follows:
```
{
    "doc_id": "",
    "sent_id": "",
    "tokens": [],
    "pieces": [],
    "token_lens": [],
    "sentence": "",
    "entity_mentions": [],
    "relation_mentions": [],
    "event_mentions": []
}
```

### (2) Merge nested annotation

Place the `ACE2005` dataset in the `data` folder, which should be in the same directory as the `nest_annotation` data. The file structure is as follows:
```
- data
    - ACE2005
        - train.oneie.json
        - dev.oneie.json
        - test.oneie.json
    - nest_annotation
        - train.json
        - dev.json
        - test.json
```  

Run the merge_annotation.py
`python merge_annotation.py`

## 2. Code for PerNee

Coming soon.
