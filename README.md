# Nested Event Extraction upon Pivot Element Recognition 

The ACE2005-Nest dataset and the code of the PerNee model for COLING 2024 paper: [Nested Event Extraction upon Pivot Element Recognition](https://arxiv.org/)


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

> The code references some of the [OneIE code](https://blender.cs.illinois.edu/software/oneie/). Thanks to the authors of OneIE.

### (1) Environments
```
- python (3.7.13)
- cuda (11.1)
```

### (2) Dependencies

`pip install -r requirements`


### (3) Training

`python train.py`

### (4) License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

### (5) Citation

If you find the dataset or paper helpful, please cite our work:

```
@inproceedings{ren-etal-2024-nested-event,
    title = "Nested Event Extraction upon Pivot Element Recognition",
    author = "Ren, Weicheng  and
      Li, Zixuan  and
      Jin, Xiaolong  and
      Bai, Long  and
      Su, Miao  and
      Liu, Yantao  and
      Guan, Saiping  and
      Guo, Jiafeng  and
      Cheng, Xueqi",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1061",
    pages = "12127--12137",
    abstract = "Nested Event Extraction (NEE) aims to extract complex event structures where an event contains other events as its arguments recursively. Nested events involve a kind of Pivot Elements (PEs) that simultaneously act as arguments of outer-nest events and as triggers of inner-nest events, and thus connect them into nested structures. This special characteristic of PEs brings challenges to existing NEE methods, as they cannot well cope with the dual identities of PEs. Therefore, this paper proposes a new model, called PerNee, which extracts nested events mainly based on recognizing PEs. Specifically, PerNee first recognizes the triggers of both inner-nest and outer-nest events and further recognizes the PEs via classifying the relation type between trigger pairs. The model uses prompt learning to incorporate information from both event types and argument roles for better trigger and argument representations to improve NEE performance. Since existing NEE datasets (e.g., Genia11) are limited to specific domains and contain a narrow range of event types with nested structures, we systematically categorize nested events in the generic domain and construct a new NEE dataset, called ACE2005-Nest. Experimental results demonstrate that PerNee consistently achieves state-of-the-art performance on ACE2005-Nest, Genia11, and Genia13. The ACE2005-Nest dataset and the code of the PerNee model are available at https://github.com/waysonren/PerNee.",
}
```