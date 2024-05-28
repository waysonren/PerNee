import torch
import torch.nn as nn
from graph import Graph
from transformers import BertModel
from util import normalize_score


def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets


def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def graphs_to_node_idxs(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs, entity_masks = [], []
    trigger_idxs, trigger_masks = [], []
    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    for graph in graphs:
        seq_entity_idxs, seq_entity_masks = [], []
        seq_trigger_idxs, seq_trigger_masks = [], []
        for entity in graph.entities:
            entity_len = entity[1] - entity[0]
            seq_entity_idxs.extend([i for i in range(entity[0], entity[1])])
            seq_entity_idxs.extend([0] * (max_entity_len - entity_len))
            seq_entity_masks.extend([1.0 / entity_len] * entity_len)
            seq_entity_masks.extend([0.0] * (max_entity_len - entity_len))
        seq_entity_idxs.extend([0] * max_entity_len * (max_entity_num - graph.entity_num))
        seq_entity_masks.extend([0.0] * max_entity_len * (max_entity_num - graph.entity_num))
        entity_idxs.append(
            seq_entity_idxs)  # [[ent_start,ent_start+1,..ent_end-1,   ent2_s,0,..,0,   ent3_s,ent3_s+1,0,..,0,]]
        entity_masks.append(seq_entity_masks)

        for trigger in graph.triggers:
            trigger_len = trigger[1] - trigger[0]
            seq_trigger_idxs.extend([i for i in range(trigger[0], trigger[1])])
            seq_trigger_idxs.extend([0] * (max_trigger_len - trigger_len))
            seq_trigger_masks.extend([1.0 / trigger_len] * trigger_len)
            seq_trigger_masks.extend([0.0] * (max_trigger_len - trigger_len))
        seq_trigger_idxs.extend([0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        seq_trigger_masks.extend([0.0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        trigger_idxs.append(seq_trigger_idxs)
        trigger_masks.append(seq_trigger_masks)

    return (
        entity_idxs, entity_masks, max_entity_num, max_entity_len,
        trigger_idxs, trigger_masks, max_trigger_num, max_trigger_len,
    )


def graphs_to_label_idxs(graphs, max_entity_num=-1, max_trigger_num=-1,
                         relation_directional=False,
                         symmetric_relation_idxs=None):
    """Convert a list of graphs to label index and mask matrices
    :param graphs (list): A list of Graph objects.
    :param max_entity_num (int) Max entity number (default = -1).
    :param max_trigger_num (int) Max trigger number (default = -1).
    """
    if max_entity_num == -1:
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
    if max_trigger_num == -1:
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
    (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask,
        batch_nested_role_idxs, batch_nested_role_mask
    ) = [[] for _ in range(10)]
    for graph in graphs:
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask, nested_role_idxs, nested_role_mask,
        ) = graph.to_label_idxs(max_entity_num, max_trigger_num,
                                relation_directional=relation_directional,
                                symmetric_relation_idxs=symmetric_relation_idxs)
        batch_entity_idxs.append(entity_idxs)
        batch_entity_mask.append(entity_mask)
        batch_trigger_idxs.append(trigger_idxs)
        batch_trigger_mask.append(trigger_mask)
        batch_relation_idxs.append(relation_idxs)
        batch_relation_mask.append(relation_mask)
        batch_role_idxs.append(role_idxs)
        batch_role_mask.append(role_mask)
        batch_nested_role_idxs.append(nested_role_idxs)
        batch_nested_role_mask.append(nested_role_mask)
    return (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask,
        batch_nested_role_idxs, batch_nested_role_mask
    )


def generate_pairwise_idxs(num1, num2):
    """Generate all pairwise combinations among entity mentions (relation) or
    event triggers and entity mentions (argument role).

    For example, if there are 2 triggers and 3 mentions in a sentence, num1 = 2,
    and num2 = 3. We generate the following vector:

    idxs = [0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4]

    Suppose `trigger_reprs` and `entity_reprs` are trigger/entity representation
    tensors. We concatenate them using:

    te_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)

    After that we select vectors from `te_reprs` using (incomplete code) to obtain
    pairwise combinations of all trigger and entity vectors.

    te_reprs = torch.gather(te_reprs, 1, idxs)
    te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)

    :param num1: trigger number (argument role) or entity number (relation)
    :param num2: entity number (relation)
    :return (list): a list of indices
    """
    idxs = []
    for i in range(num1):
        for j in range(num2):
            idxs.append(i)
            idxs.append(j + num1)
    return idxs


def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        batch_mentions.append(mentions)
    return batch_mentions


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""

    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores


class PerNee(nn.Module):
    def __init__(self,
                 config,
                 vocabs,
                 tokenizer,
                 valid_patterns=None):
        super().__init__()

        # vocabularies
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
        self.mention_type_stoi = vocabs['mention_type']
        self.entity_type_stoi = vocabs['entity_type']
        self.event_type_stoi = vocabs['event_type']
        self.relation_type_stoi = vocabs['relation_type']
        self.flat_role_type_stoi = vocabs['flat_role_type']
        self.nested_role_type_stoi = vocabs['nested_role_type']
        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i: s for s, i in self.trigger_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.event_type_itos = {i: s for s, i in self.event_type_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.trigger_label_num = len(self.trigger_label_stoi)
        self.mention_type_num = len(self.mention_type_stoi)
        self.entity_type_num = len(self.entity_type_stoi)
        self.event_type_num = len(self.event_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.flat_role_type_num = len(self.flat_role_type_stoi)
        self.nested_role_type_num = len(self.nested_role_type_stoi)
        self.valid_relation_entity = set()
        self.valid_event_nested_role = set()
        self.valid_event_flat_role = set()
        self.valid_role_entity = set()
        if valid_patterns:
            self.valid_event_flat_role = valid_patterns['event_flat_role']
            self.valid_event_nested_role = valid_patterns['event_nested_role']  # nested role
            self.valid_relation_entity = valid_patterns['relation_entity']
            self.valid_role_entity = valid_patterns['role_entity']
        self.relation_directional = config.relation_directional
        self.symmetric_relations = config.symmetric_relations
        self.symmetric_relation_idxs = {self.relation_type_stoi[r]
                                        for r in self.symmetric_relations}

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert = BertModel(bert_config)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        # local classifiers
        self.use_entity_type = config.use_entity_type
        self.binary_dim = self.bert_dim * 2
        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout
        entity_hidden_num = config.entity_hidden_num
        mention_hidden_num = config.mention_hidden_num
        event_hidden_num = config.event_hidden_num
        relation_hidden_num = config.relation_hidden_num
        role_hidden_num = config.role_hidden_num
        role_input_dim = self.binary_dim + (self.entity_type_num if self.use_entity_type else 0)
        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                          bias=linear_bias)
        self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num,
                                           bias=linear_bias)
        self.entity_type_ffn = Linears([self.bert_dim, entity_hidden_num,
                                        self.entity_type_num],
                                       dropout_prob=linear_dropout,
                                       bias=linear_bias,
                                       activation=config.linear_activation)
        self.mention_type_ffn = Linears([self.bert_dim, mention_hidden_num,
                                         self.mention_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)
        self.event_type_ffn = Linears([self.bert_dim, event_hidden_num,
                                       self.event_type_num],
                                      dropout_prob=linear_dropout,
                                      bias=linear_bias,
                                      activation=config.linear_activation)
        self.event_type_prompt = Linears([self.bert_dim,
                                          self.event_type_num],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        self.event_type_candidate = Linears([self.bert_dim,
                                             self.event_type_num],
                                            dropout_prob=linear_dropout,
                                            bias=linear_bias,
                                            activation=config.linear_activation)
        self.relation_type_ffn = Linears([self.binary_dim, relation_hidden_num,
                                          self.relation_type_num],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        self.flat_role_type_prompt = Linears([self.bert_dim, self.flat_role_type_num],
                                             dropout_prob=linear_dropout,
                                             bias=linear_bias,
                                             activation=config.linear_activation)
        self.flat_role_type_ffn = Linears([role_input_dim, role_hidden_num,
                                           self.flat_role_type_num],
                                          dropout_prob=linear_dropout,
                                          bias=linear_bias,
                                          activation=config.linear_activation)
        self.nested_role_type_prompt = Linears([self.bert_dim, self.nested_role_type_num],
                                               dropout_prob=linear_dropout,
                                               bias=linear_bias,
                                               activation=config.linear_activation)
        self.nested_role_type_ffn = Linears([self.binary_dim, role_hidden_num,
                                             self.nested_role_type_num],
                                            dropout_prob=linear_dropout,
                                            bias=linear_bias,
                                            activation=config.linear_activation)
        # decoder
        self.beam_size = config.beam_size
        self.beta_v = config.beta_v
        self.beta_e = config.beta_e
        self.entity_criteria = torch.nn.CrossEntropyLoss()
        self.event_criteria = torch.nn.CrossEntropyLoss()
        self.mention_criteria = torch.nn.CrossEntropyLoss()
        self.relation_criteria = torch.nn.CrossEntropyLoss()
        self.role_criteria = torch.nn.CrossEntropyLoss()
        self.nested_role_criteria = torch.nn.CrossEntropyLoss()
        # others
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
        self.trigger_crf = CRF(self.trigger_label_stoi, bioes=False)
        self.pad_vector = nn.Parameter(torch.randn(1, 1, self.bert_dim))
        self.distance = nn.PairwiseDistance(p=2)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.layer_norm_trg = nn.LayerNorm(self.bert_dim)
        self.layer_norm_role = nn.LayerNorm(role_input_dim)
        self.layer_norm_nested_role = nn.LayerNorm(self.binary_dim)
        self.use_prompt = config.use_prompt

    def load_bert(self, name, tokenizer, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        self.bert = BertModel.from_pretrained(name,
                                              cache_dir=cache_dir,
                                              output_hidden_states=True)
        self.bert.resize_token_embeddings(len(tokenizer))

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.use_prompt:
            nested_role_prompt_offset = torch.where(piece_idxs[0] == 102)[0][
                                            2].item() + 1  # use the third [SEP] to get the offset
        else:
            nested_role_prompt_offset = 1  # skip the [CLS]
        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)
        bert_outputs_prompt = bert_outputs[:, :nested_role_prompt_offset, :]
        bert_outputs_prompt = self.bert_dropout(bert_outputs_prompt)

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
        # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + nested_role_prompt_offset
        masks = bert_outputs.new(masks).unsqueeze(-1)
        bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
        bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
        bert_outputs = bert_outputs.sum(2)
        bert_outputs = self.bert_dropout(bert_outputs)

        return bert_outputs_prompt, bert_outputs

    def scores(self, bert_outputs, graphs, entity_types_onehot=None, predict=False):
        (
            entity_idxs, entity_masks, entity_num, entity_len,
            trigger_idxs, trigger_masks, trigger_num, trigger_len,
        ) = graphs_to_node_idxs(graphs)

        batch_size, _, bert_dim = bert_outputs.size()

        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)  # list-->tensor
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)
        entity_masks = bert_outputs.new_tensor(entity_masks)
        trigger_masks = bert_outputs.new_tensor(trigger_masks)

        # entity type scores
        entity_idxs = entity_idxs.unsqueeze(-1).expand(-1, -1,
                                                       bert_dim)  # (batch_size, max_ent_num * max_ent_len)-->(batch_size, max_ent_num * max_ent_len, bert_dim)
        entity_masks = entity_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_words = torch.gather(bert_outputs, 1, entity_idxs)  # get token entity
        entity_words = entity_words * entity_masks  # multiply weight
        entity_words = entity_words.view(batch_size, entity_num, entity_len,
                                         bert_dim)  # (batch_size, max_ent_num * max_ent_len, bert_dim)-->(batch_size, max_ent_num, max_ent_len, bert_dim)
        entity_reprs = entity_words.sum(2)  # aggregate entity span  #-->(batch_size, max_ent_num, bert_dim)
        entity_type_scores = self.entity_type_ffn(entity_reprs)  # entity cls (batch_size, max_ent_num, ent_type_num)

        # mention type scores
        mention_type_scores = self.mention_type_ffn(entity_reprs)

        # trigger type scores
        trigger_idxs = trigger_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_masks = trigger_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_words = torch.gather(bert_outputs, 1, trigger_idxs)
        trigger_words = trigger_words * trigger_masks
        trigger_words = trigger_words.view(batch_size, trigger_num, trigger_len, bert_dim)
        trigger_reprs = trigger_words.sum(2)  # aggregate trigger span  #-->(batch_size, max_evt_num, bert_dim)

        trigger_reprs = self.layer_norm_trg(trigger_reprs)
        event_type_scores = self.event_type_ffn(trigger_reprs)  # (batch_size, max_evt_num, evt_type_num)

        # relation type score
        ee_idxs = generate_pairwise_idxs(entity_num, entity_num)
        ee_idxs = entity_idxs.new(ee_idxs)
        ee_idxs = ee_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        ee_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)
        ee_reprs = torch.gather(ee_reprs, 1, ee_idxs)
        ee_reprs = ee_reprs.view(batch_size, -1, 2 * bert_dim)
        relation_type_scores = self.relation_type_ffn(ee_reprs)

        # role type score
        te_idxs = generate_pairwise_idxs(trigger_num, entity_num)
        te_idxs = entity_idxs.new(te_idxs)
        te_idxs = te_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        te_reprs = torch.cat([trigger_reprs, entity_reprs], dim=1)
        te_reprs = torch.gather(te_reprs, 1, te_idxs)
        te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)  # aggregate trigger span

        if self.use_entity_type:
            if predict:
                entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
                entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_type_scores_softmax], dim=2)
            else:
                entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_types_onehot], dim=2)

        te_reprs = self.layer_norm_role(te_reprs)
        role_type_scores = self.flat_role_type_ffn(te_reprs)  # (batch_size, max_evt_num, evt_type_num)

        # Nestedrole type score
        tt_idxs = generate_pairwise_idxs(trigger_num, trigger_num)
        tt_idxs = trigger_idxs.new(tt_idxs)
        tt_idxs = tt_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        tt_reprs = torch.cat([trigger_reprs, trigger_reprs], dim=1)
        tt_reprs = torch.gather(tt_reprs, 1, tt_idxs)
        tt_reprs = tt_reprs.view(batch_size, -1, 2 * bert_dim)
        tt_reprs = self.layer_norm_nested_role(tt_reprs)

        nested_role_type_scores = self.nested_role_type_ffn(tt_reprs)  # (batch_size, max_evt_num, evt_type_num)

        return (entity_type_scores, mention_type_scores, event_type_scores,
                relation_type_scores, role_type_scores, nested_role_type_scores)

    def forward(self, batch):
        # encoding
        bert_outputs_prompt, bert_outputs = self.encode(batch.piece_idxs,
                                                        batch.attention_masks,
                                                        batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        event_prompt_pos = batch.event_prompt_idx
        flat_role_prompt_pos = batch.flat_role_prompt_idx
        nested_role_prompt_pos = batch.nested_role_prompt_idx

        # entity type indices -> one hot
        entity_types = batch.entity_type_idxs.view(batch_size, -1)
        entity_types = torch.clamp(entity_types, min=0)
        entity_types_onehot = bert_outputs.new_zeros(*entity_types.size(), self.entity_type_num)
        entity_types_onehot.scatter_(2, entity_types.unsqueeze(-1), 1)
        # identification
        entity_label_scores = self.entity_label_ffn(
            bert_outputs)  # (batch_size, max_token_num, bert_emb)-->(batch_size, max_token_num, entity_label, num)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)

        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                     batch.entity_label_idxs,
                                                     batch.token_nums)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        trigger_label_loglik = self.trigger_crf.loglik(trigger_label_scores,
                                                       batch.trigger_label_idxs,
                                                       batch.token_nums)
        # classification
        scores = self.scores(bert_outputs, batch.graphs, entity_types_onehot)
        (
            entity_type_scores, mention_type_scores, event_type_scores,
            relation_type_scores, flat_role_type_scores, nested_role_type_scores,
        ) = scores
        event_type_scores = event_type_scores.view(-1,
                                                   self.event_type_num)  # (bsz, max_evt_num, evt_type_num)-->(bsz*evt_num, evt_type_num)
        flat_role_type_scores = flat_role_type_scores.view(-1, self.flat_role_type_num)
        nested_role_type_scores = nested_role_type_scores.view(-1, self.nested_role_type_num)

        event_loss = self.event_criteria(event_type_scores, batch.event_type_idxs)
        role_loss = self.role_criteria(flat_role_type_scores, batch.flat_role_type_idxs)
        nested_role_loss = self.nested_role_criteria(nested_role_type_scores,
                                                     batch.nested_role_type_idxs)
        classification_loss = event_loss + role_loss + nested_role_loss
        ent_idn_loss = - entity_label_loglik.mean()
        trg_idn_loss = - trigger_label_loglik.mean()
        idn_loss = ent_idn_loss + trg_idn_loss
        loss = classification_loss + idn_loss

        return loss

    def predict(self, batch):
        self.eval()

        bert_outputs_prompt, bert_outputs = self.encode(batch.piece_idxs,
                                                        batch.attention_masks,
                                                        batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        event_prompt_pos = batch.event_prompt_idx
        flat_role_prompt_pos = batch.flat_role_prompt_idx
        nested_role_prompt_pos = batch.nested_role_prompt_idx

        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = tag_paths_to_spans(trigger_label_preds,
                                      batch.token_nums,
                                      self.trigger_label_stoi)
        node_graphs = [Graph(e, t, [], [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]
        scores = self.scores(bert_outputs, node_graphs, predict=True)
        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)
        max_trigger_num = max(max(len(seq_triggers) for seq_triggers in triggers), 1)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            spans = sorted([(*i, True) for i in seq_entities] +
                           [(*i, False) for i in seq_triggers],
                           key=lambda x: (x[0], x[1], not x[-1]))
            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                batch_graphs.append(Graph.empty_graph(self.vocabs))
                continue
            graph = self.decode(spans,
                                entity_type_scores=scores[0][i],
                                mention_type_scores=scores[1][i],
                                event_type_scores=scores[2][i],
                                relation_type_scores=scores[3][i],
                                flat_role_type_scores=scores[4][i],
                                nested_role_type_scores=scores[5][i],
                                entity_num=max_entity_num,
                                trigger_num=max_trigger_num)
            batch_graphs.append(graph)

        self.train()
        return batch_graphs

    def predict_withoracle(self, batch):
        self.eval()

        bert_outputs_prompt, bert_outputs = self.encode(batch.piece_idxs,
                                                        batch.attention_masks,
                                                        batch.token_lens)
        batch_size, _, _ = bert_outputs.size()
        event_prompt_pos = batch.event_prompt_idx
        flat_role_prompt_pos = batch.flat_role_prompt_idx
        nested_role_prompt_pos = batch.nested_role_prompt_idx

        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = []
        for graph in batch.graphs:
            trg = []
            for t in graph.triggers:
                trg.append([t[0], t[1], self.trigger_label_itos[t[2]]])
            triggers.append(trg)

        node_graphs = [Graph(e, t, [], [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]
        scores = self.scores(bert_outputs, node_graphs, predict=True)
        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)
        max_trigger_num = max(max(len(seq_triggers) for seq_triggers in triggers), 1)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            spans = sorted([(*i, True) for i in seq_entities] +
                           [(*i, False) for i in seq_triggers],
                           key=lambda x: (x[0], x[1], not x[-1]))
            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                batch_graphs.append(Graph.empty_graph(self.vocabs))
                continue
            graph = self.decode(spans,
                                entity_type_scores=scores[0][i],
                                mention_type_scores=scores[1][i],
                                event_type_scores=scores[2][i],
                                relation_type_scores=scores[3][i],
                                flat_role_type_scores=scores[4][i],
                                nested_role_type_scores=scores[5][i],
                                entity_num=max_entity_num,
                                trigger_num=max_trigger_num)
            batch_graphs.append(graph)

        self.train()
        return batch_graphs

    def compute_graph_scores(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores, nested_role_type_scores
        ) = scores
        label_idxs = graphs_to_label_idxs(graphs)
        label_idxs = [entity_type_scores.new_tensor(idx,
                                                    dtype=torch.long if i % 2 == 0
                                                    else torch.float)
                      for i, idx in enumerate(label_idxs)]
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask, nested_role_idxs, nested_role_mask
        ) = label_idxs
        # Entity score
        entity_idxs = entity_idxs.unsqueeze(-1)
        entity_scores = torch.gather(entity_type_scores, 2, entity_idxs)
        entity_scores = entity_scores.squeeze(-1) * entity_mask
        entity_score = entity_scores.sum(1)
        # Trigger score
        trigger_idxs = trigger_idxs.unsqueeze(-1)
        trigger_scores = torch.gather(trigger_type_scores, 2, trigger_idxs)
        trigger_scores = trigger_scores.squeeze(-1) * trigger_mask
        trigger_score = trigger_scores.sum(1)
        # Relation score
        relation_idxs = relation_idxs.unsqueeze(-1)
        relation_scores = torch.gather(relation_type_scores, 2, relation_idxs)
        relation_scores = relation_scores.squeeze(-1) * relation_mask
        relation_score = relation_scores.sum(1)
        # Role score
        role_idxs = role_idxs.unsqueeze(-1)
        role_scores = torch.gather(role_type_scores, 2, role_idxs)
        role_scores = role_scores.squeeze(-1) * role_mask
        role_score = role_scores.sum(1)
        # NestedRole score
        nested_role_idxs = nested_role_idxs.unsqueeze(-1)
        nested_role_scores = torch.gather(nested_role_type_scores, 2, nested_role_idxs)
        nested_role_scores = nested_role_scores.squeeze(-1) * nested_role_mask
        nested_role_score = nested_role_scores.sum(1)

        score = entity_score + trigger_score + role_score + nested_role_score

        return score

    def generate_locally_top_graphs(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores, nested_role_type_scores
        ) = scores
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
        top_graphs = []
        for graph_idx, graph in enumerate(graphs):
            entity_num = graph.entity_num
            trigger_num = graph.trigger_num
            _, top_entities = entity_type_scores[graph_idx].max(1)
            top_entities = top_entities.tolist()[:entity_num]
            top_entities = [(i, j, k) for (i, j, _), k in
                            zip(graph.entities, top_entities)]
            _, top_triggers = trigger_type_scores[graph_idx].max(1)
            top_triggers = top_triggers.tolist()[:trigger_num]
            top_triggers = [(i, j, k) for (i, j, _), k in
                            zip(graph.triggers, top_triggers)]

            top_relation_scores, top_relation_labels = relation_type_scores[graph_idx].max(1)
            top_relation_scores = top_relation_scores.tolist()
            top_relation_labels = top_relation_labels.tolist()
            top_relations = [(i, j) for i, j in zip(top_relation_scores, top_relation_labels)]
            top_relation_list = []
            for i in range(entity_num):
                for j in range(entity_num):
                    if i < j:
                        score_1, label_1 = top_relations[i * max_entity_num + j]
                        score_2, label_2 = top_relations[j * max_entity_num + i]
                        if score_1 > score_2 and label_1 != 0:
                            top_relation_list.append((i, j, label_1))
                        if score_2 > score_1 and label_2 != 0:
                            top_relation_list.append((j, i, label_2))

            _, top_roles = role_type_scores[graph_idx].max(1)
            top_roles = top_roles.tolist()
            top_roles = [(i, j, top_roles[i * max_entity_num + j])
                         for i in range(trigger_num) for j in range(entity_num)
                         if top_roles[i * max_entity_num + j] != 0]

            top_nested_role_scores, top_nested_role_labels = nested_role_type_scores[graph_idx].max(1)
            top_nested_role_scores = top_nested_role_scores.tolist()
            top_nested_role_labels = top_nested_role_labels.tolist()
            top_nested_roles = [(i, j) for i, j in zip(top_nested_role_scores, top_nested_role_labels)]
            top_nested_role_list = []
            for i in range(trigger_num):
                for j in range(trigger_num):
                    if i < j:
                        score_1, label_1 = top_nested_roles[i * max_trigger_num + j]
                        score_2, label_2 = top_nested_roles[j * max_trigger_num + i]
                        if score_1 > score_2 and label_1 != 0:
                            top_nested_role_list.append((i, j, label_1))
                        if score_2 > score_1 and label_2 != 0:
                            top_nested_role_list.append((j, i, label_2))
            top_graphs.append(Graph(
                entities=top_entities,
                triggers=top_triggers,
                relations=top_relation_list,
                roles=top_roles,
                nested_roles=top_nested_role_list,
                vocabs=graph.vocabs
            ))
        return top_graphs

    def trim_beam_set(self, beam_set, beam_size):
        if len(beam_set) > beam_size:
            beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
            beam_set = beam_set[:beam_size]
        return beam_set

    def compute_graph_score(self, graph):
        score = graph.graph_local_score
        return score

    def decode(self,
               spans,
               entity_type_scores,
               mention_type_scores,
               event_type_scores,
               relation_type_scores,
               flat_role_type_scores,
               nested_role_type_scores,
               entity_num,
               trigger_num):
        beam_set = [Graph.empty_graph(self.vocabs)]
        entity_idx, trigger_idx = 0, 0

        for start, end, _, is_entity_node in spans:
            # 1. node step
            if is_entity_node:
                node_scores = entity_type_scores[entity_idx].tolist()
            else:
                node_scores = event_type_scores[trigger_idx].tolist()
            node_scores_norm = normalize_score(node_scores)
            node_scores = [(s, i, n) for i, (s, n) in enumerate(zip(node_scores,
                                                                    node_scores_norm))]
            node_scores.sort(key=lambda x: x[0], reverse=True)
            top_node_scores = node_scores[:self.beta_v]

            beam_set_ = []
            for graph in beam_set:
                for score, label, score_norm in top_node_scores:
                    graph_ = graph.copy()
                    if is_entity_node:
                        graph_.add_entity(start, end, label, 1, 1)
                    else:
                        graph_.add_trigger(start, end, label, score, score_norm)
                    beam_set_.append(graph_)
            beam_set = beam_set_

            # 2. edge step
            if is_entity_node:
                # add a new entity: new relations, new argument roles
                for i in range(entity_idx):
                    # add relation edges
                    edge_scores_1 = relation_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_2 = relation_type_scores[entity_idx * entity_num + i].tolist()
                    edge_scores_norm_1 = normalize_score(edge_scores_1)
                    edge_scores_norm_2 = normalize_score(edge_scores_2)

                    if self.relation_directional:
                        edge_scores = [(max(s1, s2), n2 if s1 < s2 else n1, i, s1 < s2)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]
                    else:
                        edge_scores = [(max(s1, s2), n2 if s1 < n2 else n1, i, False)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, score_norm, label, inverse in top_edge_scores:
                            rel_cur_ent = label * 100 + graph.entities[-1][-1]
                            rel_pre_ent = label * 100 + graph.entities[i][-1]
                            if label == 0 or (rel_pre_ent in self.valid_relation_entity and
                                              rel_cur_ent in self.valid_relation_entity):
                                graph_ = graph.copy()
                                if self.relation_directional and inverse:
                                    graph_.add_relation(entity_idx, i, label, score, score_norm)
                                else:
                                    graph_.add_relation(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_relation(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 200:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                for i in range(trigger_idx):
                    # add argument role edges
                    edge_scores = flat_role_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores, edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            role_entity = label * 100 + graph.entities[-1][-1]
                            event_role = graph.triggers[i][-1] * 100 + label
                            if label == 0 or (event_role in self.valid_event_flat_role and  # modify to flat_role
                                              role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            # graph_.add_role(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)
                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            else:
                # add a new trigger: new argument roles
                for i in range(entity_idx):
                    edge_scores = flat_role_type_scores[trigger_idx * entity_num + i].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores,
                                                                            edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            event_role = graph.triggers[-1][-1] * 100 + label
                            role_entity = label * 100 + graph.entities[i][-1]
                            if label == 0 or (event_role in self.valid_event_flat_role
                                              and role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(trigger_idx, i, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            # graph_.add_role(trigger_idx, i, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                for i in range(trigger_idx):
                    edge_scores_1 = nested_role_type_scores[i * trigger_num + trigger_idx].tolist()
                    edge_scores_2 = nested_role_type_scores[trigger_idx * trigger_num + i].tolist()
                    edge_scores_norm_1 = normalize_score(edge_scores_1)
                    edge_scores_norm_2 = normalize_score(edge_scores_2)
                    edge_scores = [(max(s1, s2), n2 if s1 < s2 else n1, i, s1 < s2)
                                   for i, (s1, s2, n1, n2)
                                   in enumerate(zip(edge_scores_1, edge_scores_2,
                                                    edge_scores_norm_1,
                                                    edge_scores_norm_2))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, score_norm, label, inverse in top_edge_scores:
                            rel_cur_trg = graph.triggers[-1][-1] * 100 + label
                            rel_pre_trg = graph.triggers[i][-1] * 100 + label
                            # if label == 0 or (rel_pre_trg in self.valid_event_nested_role or
                            #                   rel_cur_trg in self.valid_event_nested_role):
                            graph_ = graph.copy()
                            if inverse and rel_cur_trg in self.valid_event_nested_role:
                                graph_.add_nested_role(trigger_idx, i, label, score, score_norm)
                                has_valid_edge = True
                            elif not inverse and rel_pre_trg in self.valid_event_nested_role:
                                graph_.add_nested_role(i, trigger_idx, label, score, score_norm)
                                has_valid_edge = True
                            else:
                                continue
                            beam_set_.append(graph_)

                        if not has_valid_edge:
                            graph_ = graph.copy()
                            # graph_.add_nested_role(trigger_idx, i, 0, null_score)
                            # graph_.add_nested_role(i, trigger_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            if is_entity_node:
                entity_idx += 1
            else:
                trigger_idx += 1
        beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
        graph = beam_set[0]

        # predict mention types
        _, mention_types = mention_type_scores.max(dim=1)
        mention_types = mention_types[:entity_idx]
        mention_list = [(i, j, l.item()) for (i, j, k), l
                        in zip(graph.entities, mention_types)]
        graph.mentions = mention_list

        return graph
