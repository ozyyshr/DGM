import os
import shutil
import torch
import logging
import importlib
import itertools
import numpy as np
import json
import dgl
import math
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig, AdamW, get_linear_schedule_with_warmup
from argparse import Namespace
from model.transformer import TransformerEncoder, TransformerEncoderLayer
from model.gcn import GCN_layers, RGCNModel
from torch.nn.init import xavier_uniform_
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']

relation_key_pair = {'Comment': 0, 'Clarification_question': 1, 'Elaboration': 2, 'Acknowledgement': 3, 'Continuation': 4, 'Explanation': 5, 'Conditional': 6, 'Question-answer_pair': 7, 'Alternation': 8, 'Q-Elab': 9, 'Result': 10, 'Background': 11, 'Narration': 12, 'Correction': 13, 'Parallel': 14, 'Contrast': 15}


class MHA(ElectraModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = (config.hidden_size)// config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

class FuseLayer(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()

        self.linear1 = nn.Linear(4 * args.bert_hidden_size, args.bert_hidden_size)
        self.linear2 = nn.Linear(4 * args.bert_hidden_size, args.bert_hidden_size)
        self.linear3 = nn.Linear(2 * args.bert_hidden_size, args.bert_hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim = -1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim = -1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim = -1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2


class Module(nn.Module):

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.fuse = FuseLayer(args)
        self.num_decoupling = 1
        self.epoch = 0
        self.dropout = nn.Dropout(self.args.dropout)

        # Entailment Tracking
        # electra_model_path = '/home/oysr/sharc_train/electra-large-discriminator'
        electra_model_path = args.pretrained_lm_path
        tokenizer = ElectraTokenizer.from_pretrained(electra_model_path, cache_dir=None)
        electra_config = ElectraConfig.from_pretrained(electra_model_path, cache_dir=None)
        self.electra = ElectraModel.from_pretrained(electra_model_path, cache_dir=None, config=electra_config)
        tokenizer.add_special_tokens({'additional_special_tokens':["[RULE]"]})
        self.electra.resize_token_embeddings(len(tokenizer))
        self.localMHA = nn.ModuleList([MHA(electra_config) for _ in range(1)])
        self.globalMHA = nn.ModuleList([MHA(electra_config) for _ in range(1)])
        self.relation_embeds = nn.Embedding(18,args.bert_hidden_size)  
        self.edge_embeds = nn.Embedding(6, args.bert_hidden_size)
        encoder_layer = TransformerEncoderLayer(self.args.bert_hidden_size, 16, 4 * self.args.bert_hidden_size)
        encoder_norm = nn.LayerNorm(self.args.bert_hidden_size)
        # self.GCN = GCN_layers(self.args.bert_hidden_size, 2)
        self.GCN = RGCNModel(self.args.bert_hidden_size, 6, 1, True)
        self.GCN.to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layer, args.trans_layer, encoder_norm)
        self._reset_transformer_parameters()
        self.w_entail = nn.Linear(self.args.bert_hidden_size, 3, bias=True)

        # Logic Reasoning
        self.entail_emb = nn.Parameter(torch.rand(3, self.args.bert_hidden_size))
        nn.init.normal_(self.entail_emb)

        self.w_selfattn = nn.Linear(self.args.bert_hidden_size*2, 1, bias=True)
        self.w_output = nn.Linear(self.args.bert_hidden_size*2, 4, bias=True)

    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                xavier_uniform_(param)

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('model.{}'.format(name)).Module

    @classmethod
    def load(cls, fname, override_args=None):
        load = torch.load(fname, map_location=lambda storage, loc: storage)
        args = vars(load['args'])
        if override_args:
            args.update(override_args)
        args = Namespace(**args)
        model = cls.load_module(args.model)(args)
        model.load_state_dict(load['state'])
        return model

    def save(self, metrics, dsave, early_stop):
        files = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.pt') and f != 'best.pt']
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(files) > self.args.keep-1:
            for f in files[self.args.keep-1:]:
                os.remove(f)

        fsave = os.path.join(dsave, 'step{}-{}.pt'.format(metrics['global_step'], metrics[early_stop]))
        torch.save({
            'args': self.args,
            'state': self.state_dict(),  # comment to save space
            'metrics': metrics,
        }, fsave)
        fbest = os.path.join(dsave, 'best.pt')
        if os.path.isfile(fbest):
            os.remove(fbest)
        shutil.copy(fsave, fbest)

    def create_input_tensors(self, batch):
        relation_batch = [e['entail']['relations'] for e in batch]
        feat = {
            k: torch.stack([e['entail'][k] for e in batch], dim=0).to(self.device) for k in ['input_ids', 'input_mask']
        }
        scenario_idxs = [e['entail']['scenario_idx'] for e in batch]
        return feat, relation_batch, scenario_idxs

    def logic_op(self, input, input_mask):
        selfattn_unmask = self.w_selfattn(self.dropout(input))
        selfattn_unmask.masked_fill_(~input_mask, -float('inf'))
        selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        selfattn = torch.sum(selfattn_weight * input, dim=1)
        score = self.w_output(self.dropout(selfattn))
        return score

    def forward(self, batch):
        out, out_r, out_scene_idx = self.create_input_tensors(batch)
        out['electra_enc'] = electra_enc = self.electra(input_ids=out['input_ids'], attention_mask=out['input_mask'])[0]
        # transformer encoding
        tenc_input, tenc_mask, tenc_input_gcn, tenc_input_rule = [], [], [], []
        rule_mask = []
        userinfo_input, rule_input = [], []
        for idx, e in enumerate(batch):
            G = dgl.DGLGraph().to(self.device)
            relation = []
            edge_type = []    # in total six type of edges
            edges = []
            for item in out_r[idx]:
                if item['type'] not in relation:
                    relation.append(item['type'])
            G.add_nodes(e['entail']['rule_idx'].shape[0] + 1 + len(relation))       # total utterance nodes in the graph

            # Graph Construction
            for item in out_r[idx]:
                # add default_in and default_out edges
                G.add_edges(item['y'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1)
                edge_type.append(0)
                edges.append([item['y'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1])
                # G.edges[item['y'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([0,]).to(self.device)))
                G.add_edges(relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['x'])
                edge_type.append(1)
                edges.append([relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['x']])
                # G.edges[relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['x']].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([1,]).to(self.device)))
                
                # add reverse_out and reverse_in edges
                G.add_edges(relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['y'])
                edge_type.append(2)
                edges.append([relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['y']])
                # G.edges[relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['y']].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([2,]).to(self.device)))
                G.add_edges(item['x'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1)
                edge_type.append(3)
                edges.append([item['x'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1])
                # G.edges[item['x'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([3,]).to(self.device)))
            
            # add self edges
            for x in range(e['entail']['rule_idx'].shape[0] + 1 + len(relation)):
                G.add_edges(x,x)
                edge_type.append(4)
                edges.append([x,x])
                # G.edges[x,x].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([4,]).to(self.device)))
            
            # add global edges
            for x in range(e['entail']['rule_idx'].shape[0] + 1 + len(relation)):
                if x != e['entail']['rule_idx'].shape[0]:
                    G.add_edges(e['entail']['rule_idx'].shape[0], x)
                    edge_type.append(5)
                    edges.append([e['entail']['rule_idx'].shape[0], x])
                    # G.edges[x,x].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([5,]).to(self.device)))
            
            # add node feature
            for i in range(e['entail']['rule_idx'].shape[0] + 1 + len(relation)):
                if i < e['entail']['rule_idx'].shape[0]:
                    G.nodes[[i]].data['h'] = torch.index_select(electra_enc[idx], 0, torch.LongTensor([e['entail']['rule_idx'][i],]).to(self.device))
                elif i == e['entail']['rule_idx'].shape[0]:
                    if out_scene_idx[idx] != -1:
                        G.nodes[[i]].data['h'] = torch.index_select(electra_enc[idx], 0, torch.LongTensor([e['entail']['user_idx'][1],]).to(self.device))
                    else:
                        G.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.device)))
                    
                else:
                    index_relation = relation_key_pair[relation[i-e['entail']['rule_idx'].shape[0]-1]]
                    G.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.device)))
            
            edge_norm = []
            for e1, e2 in edges:
                if e1 == e2:
                    edge_norm.append(1)
                else:
                    edge_norm.append(1/(G.in_degrees(e2)-1))


            edge_type = torch.from_numpy(np.array(edge_type)).to(self.device)
            edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().cuda()
            G.edata.update({'rel_type': edge_type,})
            G.edata.update({'norm': edge_norm})
            X = self.GCN(G)[0]   # [bz, hdim]

            tenc_idx = torch.cat([e['entail']['rule_idx'], e['entail']['user_idx']], dim=-1).to(self.device)
            gcn_user = torch.index_select(electra_enc[idx], 0, e['entail']['user_idx'].to(self.device))
            gcn_rule_idx = torch.LongTensor([i for i in range(e['entail']['rule_idx'].shape[0])]).to(self.device)
            gcn_rule = torch.index_select(X, 0, gcn_rule_idx)
            tenc_input_gcn.append(torch.cat([gcn_rule, gcn_user], dim=0))


            tenc_input_ = []
            tenc_input_global = []
            inp_ = e['entail']['user_idx'][0]
            inp = e['entail']['inp']
            ruleidx = e['entail']['rule_idx']
            # construct mask matrix for multihead attention
            M1 = M2 = torch.zeros(inp_, inp_)

            for id_ in range(len(ruleidx)-1):
                M1[ruleidx[id_]:ruleidx[id_+1], ruleidx[id_]:ruleidx[id_+1]] = 1.0
            M1[ruleidx[-1]:inp_, ruleidx[-1]:inp_] = 1.0

            M2 = 1.0-M1
            M1 = (1.0-M1) * -10000.0
            M2 = (1.0-M2) * -10000.0

            M1 = M1.unsqueeze(0).unsqueeze(1)
            M2 = M2.unsqueeze(0).unsqueeze(1)

            s = [i for i in range(e['entail']['user_idx'][0])]
            s = torch.LongTensor(s)

            rule_selected = torch.index_select(electra_enc[idx],0,s.to(self.device))
            rule_selected = rule_selected.unsqueeze(0)

            local_word_level = self.localMHA[0](rule_selected, rule_selected, attention_mask = M1.to(self.device))[0]
            global_word_level = self.globalMHA[0](rule_selected, rule_selected, attention_mask = M2.to(self.device))[0]

            for t in range(1, self.num_decoupling):
                local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask = M1.to(self.device))[0]
                global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask = M2.to(self.device))[0]

            context_word_level = self.fuse(rule_selected, local_word_level, global_word_level)

            rule_input.append(torch.index_select(context_word_level.squeeze(0), 0, e['entail']['rule_idx'].to(self.device)))
            
            userinfo_input.append(torch.index_select(electra_enc[idx], 0, e['entail']['user_idx'].to(self.device)))

            for i in rule_input[-1]:
                tenc_input_.append(i)
            for j in userinfo_input[-1]:
                tenc_input_.append(j)
            tenc_input_rule.append(torch.Tensor([t.cpu().detach().numpy() for t in tenc_input_]))


            tenc_input.append(torch.index_select(electra_enc[idx], 0, tenc_idx))
            tenc_mask.append(torch.tensor([False] * tenc_idx.shape[0], dtype=torch.bool))
            rule_mask.append(torch.tensor([1] * e['entail']['rule_idx'].shape[0], dtype=torch.bool))
        if self.args.trans_layer > 0:
            tenc_input_gcn_padded = torch.nn.utils.rnn.pad_sequence(tenc_input_gcn).to(self.device)
            tenc_input_padded = torch.nn.utils.rnn.pad_sequence(tenc_input).to(self.device)  # [seqlen, N, dim]
            tenc_input_rule_padded = torch.nn.utils.rnn.pad_sequence(tenc_input_rule).to(self.device)
            tenc_mask_padded = torch.nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True, padding_value=True).to(self.device)
            tenc_out = self.transformer_encoder(tenc_input_padded, src_key_padding_mask=tenc_mask_padded)
            tenc_out_gcn = self.transformer_encoder(tenc_input_gcn_padded, src_key_padding_mask=tenc_mask_padded)
            tenc_out_rule = self.transformer_encoder(tenc_input_rule_padded, src_key_padding_mask=tenc_mask_padded)
            tenc_out_t = torch.transpose(tenc_out+tenc_out_gcn+tenc_out_rule, 0, 1).contiguous()  # [bz * seqlen * dim]
        else:
            tenc_out_t = torch.nn.utils.rnn.pad_sequence(tenc_input + tenc_input_gcn+tenc_input_rule, batch_first=True).to(self.device)

        # predict entailment score
        entail_score_mask = nn.utils.rnn.pad_sequence(rule_mask, batch_first=True).unsqueeze(-1).to(self.device)  # [bz * seqlen * 1]
        max_num_rule_in_batch = entail_score_mask.shape[1]
        out['entail_score_pred'] = self.w_entail(self.dropout(tenc_out_t[:, :max_num_rule_in_batch, :]))  # [bz * seqlen * 3]
        entail_state = torch.matmul(out['entail_score_pred'], self.entail_emb)

        cat_state = torch.cat([entail_state, tenc_out_t[:, :max_num_rule_in_batch, :]], dim=-1)
        out['clf_score'] = self.logic_op(cat_state, entail_score_mask)  # [bz, 4]
        return out

    def extract_preds(self, out, batch):
        preds = []
        clf = out['clf_score'].max(1)[1].tolist()
        for idx, ex in enumerate(batch):
            clf_i = clf[idx]
            a = DECISION_CLASSES[clf_i]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'pred_answer': a,
                'pred_answer_cls': clf_i,
                'snippet': ex['snippet']
            })
            if 'entailment_score_gold_ce' in ex['entail']:
                gold_entail = ex['entail']['entailment_score_gold_ce'].tolist()
                pred_entail = out['entail_score_pred'][idx, :, :].max(1)[1][:ex['entail']['entailment_score_gold_ce'].shape[0]].tolist()
                preds[-1]['pred_entail'] = pred_entail
                preds[-1]['gold_entail'] = gold_entail

        return preds

    def compute_loss(self, out, batch):
        gclf = torch.tensor([ex['logic']['answer_class'] for ex in batch], device=self.device, dtype=torch.long)
        gentail = torch.nn.utils.rnn.pad_sequence([ex['entail']['entailment_score_gold_ce'] for ex in batch], padding_value=-1, batch_first=True).to(self.device).view(-1)
        loss = {
            'clf': F.cross_entropy(out['clf_score'], gclf),
            'entail': F.cross_entropy(out['entail_score_pred'].view(-1, 3), gentail, ignore_index=-1) * self.args.loss_entail_weight,
        }
        return loss

    def compute_metrics(self, predictions, data):
        from sklearn.metrics import accuracy_score, confusion_matrix
        metrics = {}
        preds = [pred['pred_answer_cls'] for pred in predictions]
        golds = [gold['logic']['answer_class'] for gold in data]
        micro_accuracy = accuracy_score(golds, preds)
        metrics["0c_micro_accuracy"] = float("{0:.2f}".format(micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        conf_mat = confusion_matrix(golds, preds, labels=[0, 1, 2, 3])
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat.shape[0])])
        metrics["0b_macro_accuracy"] = float("{0:.2f}".format(macro_accuracy * 100))  # int(100 * macro_accuracy) / 100
        metrics["0a_combined"] = float("{0:.2f}".format(macro_accuracy * micro_accuracy * 100))
        metrics["0d_confmat"] = conf_mat.tolist()
        # entailment tracking
        entail_preds = [i for ex in predictions for i in ex['pred_entail']]
        entail_golds = [i for ex in predictions for i in ex['gold_entail']]
        entail_micro_accuracy = accuracy_score(entail_golds, entail_preds)
        metrics["1b_entail_micro_accuracy"] = float("{0:.2f}".format(entail_micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        entail_conf_mat = confusion_matrix(entail_golds, entail_preds, labels=[0, 1, 2])
        entail_conf_mat_norm = entail_conf_mat.astype('float') / entail_conf_mat.sum(axis=1)[:, np.newaxis]
        entail_macro_accuracy = np.mean([entail_conf_mat_norm[i][i] for i in range(entail_conf_mat.shape[0])])
        metrics["1a_entail_macro_accuracy"] = float("{0:.2f}".format(entail_macro_accuracy * 100))  # int(100 * macro_accuracy) / 100

        return metrics

    def run_pred(self, dev):
        preds = []
        self.eval()
        for i in trange(0, len(dev), self.args.dev_batch, desc='batch', disable=self.args.tqdm_bar):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            preds += self.extract_preds(out, batch)
        return preds

    def run_train(self, train, dev):
        if not os.path.isdir(self.args.dsave):
            os.makedirs(self.args.dsave)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args.dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)

        num_train_steps = int(len(train) / self.args.train_batch * self.args.epoch)
        num_warmup_steps = int(self.args.warmup * num_train_steps)

        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)  # PyTorch scheduler

        print('num_train', len(train))
        print('num_dev', len(dev))

        global_step = 0
        best_metrics = {self.args.early_stop: -float('inf')}
        for epoch in trange(self.args.epoch, desc='epoch',):
            self.epoch = epoch
            train = train[:]
            np.random.shuffle(train)

            train_stats = defaultdict(list)
            preds = []
            self.train()
            for i in trange(0, len(train), self.args.train_batch, desc='batch'):
                actual_train_batch = int(self.args.train_batch / self.args.gradient_accumulation_steps)
                batch_stats = defaultdict(list)
                batch = train[i: i + self.args.train_batch]

                for accu_i in range(0, len(batch), actual_train_batch):
                    actual_batch = batch[accu_i : accu_i + actual_train_batch]
                    out = self(actual_batch)
                    pred = self.extract_preds(out, actual_batch)
                    loss = self.compute_loss(out, actual_batch)

                    for k, v in loss.items():
                        loss[k] = v / self.args.gradient_accumulation_steps
                        batch_stats[k].append(v.item()/ self.args.gradient_accumulation_steps)
                    sum(loss.values()).backward()
                    preds += pred

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                for k in batch_stats.keys():
                    train_stats['loss_' + k].append(sum(batch_stats[k]))

                if global_step % self.args.eval_every_steps == 0:
                    dev_stats = defaultdict(list)
                    dev_preds = self.run_pred(dev)
                    dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
                    dev_metrics.update(self.compute_metrics(dev_preds, dev))
                    metrics = {'global_step': global_step}
                    metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
                    logger.critical(pformat(metrics))

                    if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                        logger.critical('Found new best! Saving to ' + self.args.dsave)
                        best_metrics = metrics
                        self.save(best_metrics, self.args.dsave, self.args.early_stop)
                        with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                            json.dump(dev_preds, f, indent=2)
                        with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                            json.dump(best_metrics, f, indent=2)

                    self.train()

            train_metrics = {k: sum(v) / len(v) for k, v in train_stats.items()}
            train_metrics.update(self.compute_metrics(preds, train))

            dev_stats = defaultdict(list)
            dev_preds = self.run_pred(dev)
            dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
            dev_metrics.update(self.compute_metrics(dev_preds, dev))
            metrics = {'global_step': global_step}
            metrics.update({'train_' + k: v for k, v in train_metrics.items()})
            metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
            logger.critical(pformat(metrics))

            if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                logger.critical('Found new best! Saving to ' + self.args.dsave)
                best_metrics = metrics
                self.save(best_metrics, self.args.dsave, self.args.early_stop)
                with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                    json.dump(dev_preds, f, indent=2)
                with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                    json.dump(best_metrics, f, indent=2)

        logger.critical('Best dev')
        logger.critical(pformat(best_metrics))