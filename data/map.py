"""
Modified from HERO implementation
(https://github.com/linjieli222/HERO)

MAP dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd
import copy
import random

import os
import json
from .data import (VideoFeatSingleSubTokDataset, pad_tensors, get_gather_index, _check_ngpu)
from .mam import (VideoMamDataset, mam_collate, create_mam_io, _create_v_type_ids, _get_txt_tgt_mask)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


class MAPFullEvalDataset(VideoMamDataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=1.0, mask_strategy="verb_only",
                 subtitle_only=False, distributed=False, for_roberta=False, without_aft=False):
        assert isinstance(vid_sub_db, VideoFeatSingleSubTokDataset)

        self.vid_sub_db = vid_sub_db
        if distributed:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids
            
        self.mask_prob = mask_prob
        self.mask_strategy = mask_strategy
        self.subtitle_only = subtitle_only
        self.for_roberta = for_roberta
        self.without_aft = without_aft

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]
        vid_seg_id = example['v_seg_id']
        verb = example['verb']
        orig_input_id = copy.deepcopy(example['input_id'])

        input_ids = None
        txt_labels = None
        output = []
        if self.for_roberta:
            is_verb = example["is_verb"]

            input_ids, txt_labels = create_mam_io(
                orig_input_id, is_verb, self.vid_sub_db.txt_db,
                self.mask_prob, self.mask_strategy)

            txt_labels_tensor = torch.tensor(txt_labels)
            txt_mask_tgt = (txt_labels_tensor != -1)
            
            # attention masks
            attn_masks = torch.ones(len(input_ids), dtype=torch.long)
            
            # tensorize
            input_ids = torch.tensor(input_ids)
            
            output.append((vid_seg_id, verb, input_ids, attn_masks, txt_mask_tgt, txt_labels))
        else:
            if self.mask_strategy == 'verb_only':
                # text input when masking verb
                # a dict to indicate whether the subtoken belongs to a verb class
                is_verb = example["is_verb"]
                
                input_ids, txt_labels = create_mam_io(
                    orig_input_id, is_verb, self.vid_sub_db.txt_db,
                    self.mask_prob, self.mask_strategy)
                
            elif self.mask_strategy == 'noun_only':
                # text input when masking noun
                # this is just for getting the noun representation for analysis
                nouns = example['nouns']
                is_noun = {}
                for token_id in orig_input_id:
                    t = tokenizer.convert_ids_to_tokens(token_id)
                    str = tokenizer.convert_tokens_to_string(t).replace(" ", "")
                    if str in nouns:
                        is_noun[token_id] = 1
                    else:
                        is_noun[token_id] = 0
                        
                input_ids, txt_labels = create_mop_io(
                    orig_input_id, is_noun, self.vid_sub_db.txt_db)
    
            # video input
            v_feat, nframes = self.vid_sub_db._get_v_feat(vid)
            
            # divide video frames into [BEF], [ACT], [AFT] video type ids
            # {[BEF]: 0, [ACT]: 1, [AFT]: 2, [PAD]: 3}
            clip_level_v_type_ids, num_AFT_frames = _create_v_type_ids(nframes)
            clip_level_v_type_ids = torch.tensor(clip_level_v_type_ids).long()
            
            # attention masks
            if self.subtitle_only:
                # mask visual part
                t_attn_masks = torch.ones(len(input_ids), dtype=torch.long)
                v_attn_masks = torch.zeros(nframes, dtype=torch.long)
                attn_masks = torch.cat((t_attn_masks, v_attn_masks), 0)
            elif self.without_aft:
                # mask [AFT] visual part
                t_attn_masks = torch.ones(len(input_ids), dtype=torch.long)
                v_attn_masks = torch.ones((nframes - num_AFT_frames), dtype=torch.long)
                v_attn_AFT_masks = torch.zeros(num_AFT_frames, dtype=torch.long)
                attn_masks = torch.cat((t_attn_masks, v_attn_masks, v_attn_AFT_masks), 0)
            else:
                attn_masks = torch.ones(len(input_ids) + nframes, dtype=torch.long)
            txt_mask_tgt = _get_txt_tgt_mask(txt_labels != -1, nframes)
            
            # tensorize
            input_ids = torch.tensor(input_ids)
    
            output.append((vid_seg_id, verb, input_ids, v_feat, clip_level_v_type_ids, attn_masks, txt_mask_tgt, txt_labels))
        return output


def noun_word(tokens, vocab_range, mask, tokens_is_noun):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        if tokens_is_noun[token] == 1:
            prob = random.random()

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def create_mop_io(input_ids, input_id2is_noun, db, cls_tok=True):
    input_ids, txt_labels = noun_word(
        input_ids, db.v_range, db.mask, input_id2is_noun)
    if cls_tok:
        input_ids = [db.cls_] + input_ids
    else:
        input_ids = [db.sep] + input_ids
    txt_labels = torch.tensor([-1] + txt_labels)
    return input_ids, txt_labels


def map_full_eval_collate_roberta(inputs):
    """
	Return:
	:vid_seg_ids  (n, vid_seg_id)
    :verbs        (n, verb)
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
	:attn_masks   (n, max_L) padded with 0
	"""
    
    (vid_seg_ids, verbs, input_ids, attn_masks, txt_masks, txt_labels) = \
        map(list, unzip(concat(inputs)))
    
    # text batches
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    txt_mask_tgt = pad_sequence(txt_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)

    # attn masks
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    
    batch = {'vid_seg_ids': vid_seg_ids,
             'verbs': verbs,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'txt_mask_tgt': txt_mask_tgt,
             'txt_labels': txt_labels[txt_labels != -1]
             }
    return batch


def map_full_eval_collate(inputs):
    """
    Return:
    :vid_seg_ids  (n, vid_seg_id)
    :verbs        (n, verb)
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :img_feat     (n, max_num_frame, feat_dim)
    :img_pos_ids  (n, max_num_frame)
    :img_type_ids     (n, max_num_frame) padded with 3
    :attn_masks   (n, max_{L + num_frames}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (vid_seg_ids, verbs, input_ids, v_feats, clip_level_v_type_ids, attn_masks, txt_masks, txt_labels
     ) = map(list, unzip(concat(inputs)))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_mask_tgt = pad_sequence(txt_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    # image batches
    num_fs = [f.size(0) for f in v_feats]
    v_feat = pad_tensors(v_feats, num_fs)

    # hard_coded padding value
    clip_level_v_type_ids = pad_sequence(
        clip_level_v_type_ids, batch_first=True, padding_value=3)

    clip_level_pos_ids = torch.arange(0, v_feat.size(1), dtype=torch.long)\
        .unsqueeze(0).expand(v_feat.size(0), -1).clone()
    
    # attn masks
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_vl, _ = v_feat.size()
    out_size = attn_masks.size(1)
    if max_vl > 0:
        gather_index = get_gather_index(txt_lens, num_fs, bs, max_vl, out_size)
    else:
        gather_index = None
        v_feat = None

    batch = {'vid_seg_ids': vid_seg_ids,
             'verbs': verbs,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'v_feat': v_feat,
             'c_v_type_ids': clip_level_v_type_ids,
             'c_v_pos_ids': clip_level_pos_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_mask_tgt': txt_mask_tgt,
             'txt_labels': txt_labels[txt_labels != -1]}
    return batch

