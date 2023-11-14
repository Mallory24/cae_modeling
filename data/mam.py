"""
Modified from HERO implementation
(https://github.com/linjieli222/HERO)

Pretrain MAM (Masked Action Modeling) dataset
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd
import copy

from .data import (VideoFeatSingleSubTokDataset,
                   pad_tensors, get_gather_index, _check_ngpu)


def random_word(tokens, vocab_range, mask, mask_prob=0.15):
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
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

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


def verb_only(tokens, vocab_range, mask, tokens_is_verb):
    """
    Masking verb tokens ONLY for masked action prediction task.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :param mask: <mask> token
    :param tokens_is_verb: a dict of token_id to 0 or 1 to indicate whether the token_id belongs to the verb
    :return: (list of int, list of int), masked tokens and related labels for
        masked action prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        if tokens_is_verb[token] == 1:
            prob = random.random()
            # 80% randomly change token to <mask> token
            if prob < 0.8:
                tokens[i] = mask
    
            # 15% randomly change token to random token
            elif prob < 0.95:
                tokens[i] = random.choice(list(range(*vocab_range)))
    
            # -> rest 5% randomly keep current token
            
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


def verb_random_joint(tokens, vocab_range, mask, tokens_is_verb, mask_prob):
    """
        Masking verb tokens and random tokens jointly with 15% probability for masked action prediction task.
        :param tokens: list of int, tokenized sentence.
        :param vocab_range: for choosing a random word
        :param mask: <mask> token
        :param tokens_is_verb: a dict of token_id to 0 or 1 to indicate whether the token_id belongs to the verb
        :param mask_prob: masking probability for the rest of the tokens
        :return: (list of int, list of int), masked tokens and related labels for
            masked action prediction
        """
    output_label = []

    for i, token in enumerate(tokens):
        if tokens_is_verb[token] == 1:
            verb_prob = random.random()
            # 80% randomly change token to <mask> token
            if verb_prob < 0.8:
                tokens[i] = mask

            # 15% randomly change token to random token
            elif verb_prob < 0.95:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 5% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            prob = random.random()
            # mask token with 15% probability
            if prob < mask_prob:
                prob /= mask_prob
            
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


def verb_random_alter(tokens, vocab_range, mask, tokens_is_verb, mask_prob):
    """
        Masking verb tokens and random tokens jointly with 15% probability for masked action prediction task.
        :param tokens: list of int, tokenized sentence.
        :param vocab_range: for choosing a random word
        :param mask: <mask> token
        :param tokens_is_verb: a dict of token_id to 0 or 1 to indicate whether the token_id belongs to the verb
        :param mask_prob: masking probability for the rest of the tokens
        :return: (list of int, list of int), masked tokens and related labels for
            masked action prediction
        """
    output_label = []
    strategy = random.choice(["verb", "random"])

    # random strategy still includes the chance of having the verb being masked
    if strategy == "verb":
        tokens, output_label = verb_only(tokens, vocab_range, mask, tokens_is_verb)
    elif strategy == "random":
        tokens, output_label = random_word(tokens, vocab_range, mask, mask_prob)
   
    return tokens, output_label


def _get_txt_tgt_mask(txt_mask, n_frame):
    z = torch.zeros(n_frame, dtype=torch.bool)
    txt_mask_tgt = torch.cat([z, txt_mask], dim=0)
    return txt_mask_tgt


def create_mlm_io(input_ids, db, mask_prob, cls_tok=True):
    input_ids, txt_labels = random_word(
        input_ids, db.v_range, db.mask, mask_prob)
    if cls_tok:
        input_ids = [db.cls_] + input_ids
    else:
        input_ids = [db.sep] + input_ids
    txt_labels = torch.tensor([-1] + txt_labels)
    return input_ids, txt_labels


def create_mam_io(input_id, input_id2is_verb, db, mask_prob, strategy="verb_only", cls_tok=True):
    # original text1: chop the carrot into pieces
    # original text2: chop the garlic into small chunks
    txt_labels = None
    if strategy == "verb_only":
        # result1: <mask> the carrot into pieces
        # result2: <mask> the garlic into small chunks
        input_id, txt_labels = verb_only(
            input_id, db.v_range, db.mask, input_id2is_verb)
    elif strategy == "verb_random_joint":
        # result1: <mask> the <mask> into pieces
        # result2: <mask> the garlic into <mask> chunks
        input_id, txt_labels = verb_random_joint(
            input_id, db.v_range, db.mask, input_id2is_verb, mask_prob)
    elif strategy == "verb_random_alter":
        # result1: <mask> the carrot into pieces
        # result2: chop the garlic into <mask> chunks
        input_id, txt_labels = verb_random_alter(
            input_id, db.v_range, db.mask, input_id2is_verb, mask_prob)
    if cls_tok:
        input_id = [db.cls_] + input_id
    else:
        input_id = [db.sep] + input_id
    txt_labels = torch.tensor([-1] + txt_labels)
    return input_id, txt_labels


def _create_v_type_ids(nframes):
    div = nframes // 3
    clip_level_v_type_ids = [0] * div + [1] * div + [2] * div
    if len(clip_level_v_type_ids) < nframes:
        rest_num = nframes - len(clip_level_v_type_ids)
        clip_level_v_type_ids += [2] * rest_num
    num_AFT_frames = clip_level_v_type_ids.count(2)
    return clip_level_v_type_ids, num_AFT_frames


class VideoMamDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15,
                 sub_ctx_len=0, mask_strategy="verb_only", subtitle_only=False, without_aft=False):
        # mask_strategy = ["verb_only", "verb_random_joint", "verb_random_alter"]
        
        assert isinstance(vid_sub_db, VideoFeatSingleSubTokDataset)
        self.mask_prob = mask_prob
        self.vid_sub_db = vid_sub_db
        if _check_ngpu() > 1:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids
        self.sub_ctx_len = sub_ctx_len  # TO CHECK: Might not need it
        self.mask_strategy = mask_strategy
        self.subtitle_only = subtitle_only
        self.without_aft = without_aft

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]
        orig_input_id = copy.deepcopy(example['input_id'])
        
        # a dict to indicate whether the subtoken belongs to a verb class
        is_verb = example["is_verb"]
        output = []
        
        # text input
        input_id, txt_label = create_mam_io(
            orig_input_id, is_verb, self.vid_sub_db.txt_db,
            self.mask_prob, self.mask_strategy)

        input_id = torch.tensor(input_id)

        # video input
        v_feat, nframes = self.vid_sub_db._get_v_feat(vid)

        # divide video frames into [BEF], [ACT], [AFT] video type ids
        # {[BEF]: 0, [ACT]: 1, [AFT]: 2, [PAD]: 3}
        clip_level_v_type_ids, num_AFT_frames = _create_v_type_ids(nframes)
        clip_level_v_type_ids = torch.tensor(clip_level_v_type_ids).long()
        
        # attention masks
        if self.subtitle_only:
            # mask visual part
            t_attn_masks = torch.ones(len(input_id), dtype=torch.long)
            v_attn_masks = torch.zeros(nframes, dtype=torch.long)
            attn_masks = torch.cat((t_attn_masks, v_attn_masks), 0)
        elif self.without_aft:
            # mask [AFT] visual part
            t_attn_masks = torch.ones(len(input_id), dtype=torch.long)
            v_attn_masks = torch.ones((nframes-num_AFT_frames), dtype=torch.long)
            v_attn_AFT_masks = torch.zeros(num_AFT_frames, dtype=torch.long)
            attn_masks = torch.cat((t_attn_masks, v_attn_masks, v_attn_AFT_masks), 0)
        else:
            attn_masks = torch.ones(len(input_id) + nframes, dtype=torch.long)
        txt_mask_tgt = _get_txt_tgt_mask(txt_label != -1, nframes)
        
        output.append((input_id, v_feat, clip_level_v_type_ids, attn_masks, txt_mask_tgt, txt_label))
        return output


def mam_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :img_feat     (n, max_num_frame, feat_dim)
    :img_pos_ids  (n, max_num_frame)
    :img_type_ids     (n, max_num_frame) padded with 3
    :attn_masks   (n, max_{L + num_frames}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, v_feats, clip_level_v_type_ids, attn_masks, txt_masks, txt_labels
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
    
    clip_level_pos_ids = torch.arange(
        0, v_feat.size(1), dtype=torch.long
    ).unsqueeze(0).expand(v_feat.size(0), -1).clone()

    # attn masks
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_vl, _ = v_feat.size()
    out_size = attn_masks.size(1)
    if max_vl > 0:
        gather_index = get_gather_index(txt_lens, num_fs, bs, max_vl, out_size)
    else:
        gather_index = None
        v_feat = None

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'v_feat': v_feat,
             'c_v_type_ids': clip_level_v_type_ids,
             'c_v_pos_ids': clip_level_pos_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_mask_tgt': txt_mask_tgt,
             'txt_labels': txt_labels[txt_labels != -1]}
    return batch
