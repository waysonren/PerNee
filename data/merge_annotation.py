import os
import json

ann_dir = "nest_annotation"
ace_dir = "ACE2005"
ace_nest_dir = "ACE2005-Nest"
if not os.path.exists(ace_nest_dir):
    os.mkdir(ace_nest_dir)

ace_datasets = ["train.oneie.json", "dev.oneie.json", "test.oneie.json"]
nest_ann_datasets = ["train.json", "dev.json", "test.json"]

for ace_dataset, nest_ann in zip(ace_datasets, nest_ann_datasets):

    ace_file = os.path.join(ace_dir, ace_dataset)
    ann_file = os.path.join(ann_dir, nest_ann)
    ace_nest_file = os.path.join(ace_nest_dir, nest_ann)
    with open(ace_file) as f_ace, open(ann_file) as f_ann, open(ace_nest_file, "w") as f_ace_nest:
        anns = json.load(f_ann)
        for raw_line in f_ace:
            line = json.loads(raw_line)
            line["entity_mentions"] = anns[line["sent_id"]]["entity_mentions"]
            line["event_mentions"] = anns[line["sent_id"]]["event_mentions"]
            line["relation_mentions"] = []
            f_ace_nest.write(json.dumps(line) + '\n')
