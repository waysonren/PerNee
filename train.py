import os
import json
import time
from argparse import ArgumentParser
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AdamW, get_linear_schedule_with_warmup)
from model import PerNee
from graph import Graph
from scorer import score_graphs
from config import Config
from data import IEDataset
from util import generate_vocabs, load_valid_patterns_nested, save_result, add_nested_vocabs, best_score
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/ace2005-nest.json')
args = parser.parse_args()
config = Config.from_json_file(args.config)

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
if config.continue_train:
    output_dir = config.pretrain_model.replace("/best.role.mdl", "")
    start_epoch = config.start_epoch
else:
    output_dir = os.path.join(config.log_path, timestamp)
    start_epoch = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')

with open(log_file, 'a', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')

# datasets
model_name = config.bert_model_name
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)
tokenizer.add_special_tokens(
    {"additional_special_tokens": config.prompt_token})

train_set = IEDataset(config.train_file, max_length=config.max_length, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title,
                      event_identity_only=config.event_identity_only, use_prompt=config.use_prompt,
                      nested_mask=config.nested_mask)
dev_set = IEDataset(config.dev_file, max_length=config.max_length, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations,
                    event_identity_only=config.event_identity_only, use_prompt=config.use_prompt,
                    nested_mask=config.nested_mask)
test_set = IEDataset(config.test_file, max_length=config.max_length, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations,
                     event_identity_only=config.event_identity_only, use_prompt=config.use_prompt,
                     nested_mask=config.nested_mask)
if config.continue_train:
    map_location = 'cuda:{}'.format(config.gpu_device) if use_gpu else 'cpu'
    state = torch.load(config.pretrain_model, map_location=map_location)
    vocabs = state['vocabs']
else:
    vocabs = generate_vocabs([train_set, dev_set, test_set])
    vocabs = add_nested_vocabs(vocabs)

f_label_mapping = open(config.trigger_representation_file)
f_label_mapping_json = json.load(f_label_mapping)
vocabs["label_mapping"] = f_label_mapping_json

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)
valid_patterns = load_valid_patterns_nested(config.valid_pattern_path, vocabs)

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
                (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
                 (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = PerNee(config, vocabs, tokenizer, valid_patterns)
model.load_bert(model_name, tokenizer, cache_dir=config.bert_cache_dir)

if config.continue_train:
    model.load_state_dict(state['model'])
if use_gpu:
    model.cuda(device=config.gpu_device)

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and 'crf' not in n and 'global_feature' not in n],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and ('crf' in n or 'global_feature' in n)],
        'lr': config.learning_rate, 'weight_decay': 0
    }
]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)

tasks = ['AC']
best_dev = {k: 0 for k in tasks}

best_dev_role_score = 0
for epoch in range(start_epoch, config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

        loss = model(batch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            global_step += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    if epoch == 0 or (epoch + 1) % config.record_step == 0:
        # dev set
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                             desc='Dev {}'.format(epoch))
        best_dev_role_model = False
        dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            graphs = model.predict(batch)
            if config.ignore_first_header:
                for inst_idx, sent_id in enumerate(batch.sent_ids):
                    if int(sent_id.split('-')[-1]) < 4:
                        graphs[inst_idx] = Graph.empty_graph(vocabs)
            for graph in graphs:
                graph.clean(relation_directional=config.relation_directional,
                            symmetric_relations=config.symmetric_relations)
            dev_gold_graphs.extend(batch.graphs)
            dev_pred_graphs.extend(graphs)
            dev_sent_ids.extend(batch.sent_ids)
            dev_tokens.extend(batch.tokens)
        progress.close()
        dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs)
        for task in tasks:
            if dev_scores[task]['f'] > best_dev[task]:
                best_dev[task] = dev_scores[task]['f']
                if task == 'AC':
                    print('Saving best role model')
                    torch.save(state, best_role_model)
                    best_dev_role_model = True
                    save_result(dev_result_file,
                                dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                                dev_tokens)

        # test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                             desc='Test {}'.format(epoch))
        test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
        for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                                collate_fn=test_set.collate_fn):
            progress.update(1)
            graphs = model.predict(batch)
            if config.ignore_first_header:
                for inst_idx, sent_id in enumerate(batch.sent_ids):
                    if int(sent_id.split('-')[-1]) < 4:
                        graphs[inst_idx] = Graph.empty_graph(vocabs)
            for graph in graphs:
                graph.clean(relation_directional=config.relation_directional,
                            symmetric_relations=config.symmetric_relations)
            test_gold_graphs.extend(batch.graphs)
            test_pred_graphs.extend(graphs)
            test_sent_ids.extend(batch.sent_ids)
            test_tokens.extend(batch.tokens)
        progress.close()
        test_scores = score_graphs(test_gold_graphs, test_pred_graphs)

        if best_dev_role_model:
            save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                        test_sent_ids, test_tokens)
            save_obj(test_gold_graphs, os.path.join(output_dir, 'test_gold_graphs'))
            save_obj(test_pred_graphs, os.path.join(output_dir, 'test_pred_graphs'))
            save_obj(test_sent_ids, os.path.join(output_dir, 'test_sent_ids'))
            save_obj(test_tokens, os.path.join(output_dir, 'test_tokens'))

        result = json.dumps(
            {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
        with open(log_file, 'a', encoding='utf-8') as w:
            w.write(result + '\n')
        print('Log file', log_file)

best_score(log_file, 'AC')
