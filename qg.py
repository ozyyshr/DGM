import os
import re
import torch
import json
from pprint import pprint
from argparse import ArgumentParser
from tqdm import tqdm
from unilmqg.biunilm.decode_seq2seq import main as qg_s2s


def preprocess_qg(preds, data):
    id2ex = {}
    for ex in data:
        id2ex[ex['utterance_id']] = ex
    qg_data = []
    lines = []
    for ex in preds:
        snippet = id2ex[ex['utterance_id']]['snippet']
        src_pred_i = ' '.join([snippet, '[SEP]', ex['answer']]).replace('\n', ' ').strip()
        ex = {
            'utterance_id': ex['utterance_id'],
            'src': src_pred_i,
        }
        qg_data.append(ex)
        lines.append(src_pred_i)
    return qg_data, lines


def preprocess_qg_(preds, span_pred):
    id2ex = {}
    for ex in span_pred:
        id2ex[ex['utterance_id']] = ex
    data = []
    lines = []
    for pred in preds:
        if pred['pred_answer'].lower() not in {'yes', 'no', 'irrelevant'}:
            if pred['utterance_id'] in id2ex.keys():
                answer = id2ex[pred['utterance_id']]['answer']
            else:
                answer = ""
            src_pred_i = ' '.join([pred['snippet'], '[SEP]', answer]).replace('\n', ' ').strip()
            ex = {
                'utterance_id': pred['utterance_id'],
                'src': src_pred_i,
            }
            data.append(ex)
            lines.append(src_pred_i)
    return data, lines

def merge_edits(preds, qgpreds):
    # note: this happens in place
    qg = {p['utterance_id']: p for p in qgpreds}
    for p in preds:
        # p['orig_answer'] = p['answer']
        if p['utterance_id'] in qg:
            p['answer'] = qg[p['utterance_id']]['answer']
        else:
            p['answer'] = p['pred_answer']
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fin', default='', help='input data file')
    parser.add_argument('--fpred', default='', help='input data file')
    parser.add_argument('--dmpred', default='', help='input dm data file')
    # parser.add_argument('--dout', default='', help='directory to store output files')
    parser.add_argument('--device', default='cuda', help='cpu not supported')
    # copy from unilm
    parser.add_argument("--bert_model", default='bert-large-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default='', type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--cache_path", default='', type=str,
                        help="Yifan added, bert vocab path")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', default=True,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=48,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()

    with open(args.fin) as f:
        sharc_data = json.load(f)
    with open(args.fpred + '/dev.preds.json') as f:
        sharc_pred = json.load(f)
    with open(args.dmpred + '/dev.preds.json') as f:
        dm_preds = json.load(f)

    ids = [ex['utterance_id'] for ex in sharc_pred]
    ids_dm = [ex['utterance_id'] for ex in dm_preds]
    sharc_data_filtered = []
    for ex in sharc_data:
        if ex['utterance_id'] in ids:
            sharc_data_filtered.append(ex)
    assert len(sharc_data_filtered) == len(sharc_pred)
	

    # qg_data, input_lines = preprocess_qg(sharc_pred, sharc_data_filtered)
    qg_data, input_lines = preprocess_qg_(dm_preds, sharc_pred)
    print('qg_data {}, input_lines {}'.format(len(qg_data), len(input_lines)))
    output_lines = qg_s2s(opt=args, inputs=input_lines)
    print("output_lines {}".format(len(output_lines)))
    qg_preds = []
    for ex, input_line, output_line in zip(qg_data, input_lines, output_lines):
        assert ex['src'] == input_line
        ex['answer'] = output_line
        qg_preds.append(ex)
    e2e_preds = merge_edits(dm_preds, qg_preds)

    with open('final_res.json', 'wt') as f:
        json.dump(e2e_preds, f, indent=2)
