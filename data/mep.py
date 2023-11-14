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
from .data import (VideoFeatSingleSubTokDataset, pad_tensors, get_gather_index, _check_ngpu, video_collate_clip_level)
from .mem import (MemDataset, mem_collate, _get_feat_target, _mask_img_feat, _get_img_mask, _create_v_type_ids)


class MEPFullEvalDataset(MemDataset):
    # mask_strategy of AFT-only => mask single frame of AFT-only for model to predict
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15, vid_type_size=4,
                 mask_strategy="AFT-only", video_frame_only=False, distributed=False, default_sampler=True):
        assert isinstance(vid_sub_db, VideoFeatSingleSubTokDataset)
        self.mask_prob = mask_prob
        self.vid_sub_db = vid_sub_db
        self.vid_type_size = vid_type_size - 1  # disregard [PAD]
        self.mask_strategy = mask_strategy
        self.video_frame_only = video_frame_only

        if distributed and default_sampler:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]
        vid_seg_id = example['v_seg_id']
        verb = example['verb']
        
        # text input
        input_id = copy.deepcopy(example['input_id'])
        input_id = torch.tensor(input_id)
        
        # video input
        c_v_feats, nframes = self.vid_sub_db._get_v_feat(vid)

        # attention masks
        if self.video_frame_only:
            # mask textual part
            t_attn_masks = torch.zeros(len(input_id), dtype=torch.long)
            v_attn_masks = torch.ones(nframes, dtype=torch.long)
            attn_masks = torch.cat((t_attn_masks, v_attn_masks), 0)
        else:
            attn_masks = torch.ones(len(input_id) + nframes, dtype=torch.long)

        # For temporal encoding of video frames
        c_v_attn_masks = torch.ones(len(c_v_feats), dtype=torch.long)
        c_v_attn_masks = torch.tensor(c_v_attn_masks)

        # divide video frames into [BEF], [ACT], [AFT] video type ids
        # {[BEF]: 0, [ACT]: 1, [AFT]: 2, [PAD]: 3}
        clip_level_v_type_ids = _create_v_type_ids(nframes)
        clip_level_v_type_ids = torch.tensor(clip_level_v_type_ids).long()

        c_frame_mask = _get_img_mask(self.mask_prob, c_v_feats.size(0), clip_level_v_type_ids,
                                     self.mask_strategy)
        
        c_pos_ids = torch.tensor(range(len(c_v_feats)), dtype=torch.long)
        c_frame_mask = c_frame_mask.index_select(0, c_pos_ids)
        
        # meta info
        vid_v_frames = [vid_seg_id + "-" + v_frame_type for v_frame_type in clip_level_v_type_ids]
        print(vid_v_frames)
        
        return (input_id, c_v_feats, clip_level_v_type_ids, attn_masks, c_v_attn_masks), \
               c_frame_mask, vid_seg_id, verb, vid_v_frames


def mep_full_eval_collate(inputs):
    """
    Return:
    :vid_seg_ids  (n, vid_seg_id)
    :verbs        (n, verb)
    :input_ids    (n, max_L) padded with 0
    :pos_ids (n, max_L) padded with 0
    :img_feat     (n, max_num_frame, feat_dim)
    :img_pos_ids (n, max_num_frame) padded with 0
    :img_type_ids     (n, max_num_frame) padded with 4
    :img_masks  (n, max_num_frame) padded with 0
    :attn_masks   (n, max_{L + num_frames}) padded with 0
    :img_attn_masks   (n, max_num_frames) padded with 0
    :img_feat_target   (n, max_num_masked_frames) padded with 0
    """
    video_inputs, c_frame_masks, vid_seg_ids, verbs, vid_v_frames = map(list, unzip(inputs))

    batch = video_collate_clip_level(video_inputs)

    # mask features
    c_frame_masks = pad_sequence(c_frame_masks,
                                 batch_first=True, padding_value=0)
    c_v_feats = batch['c_v_feats']
    feat_targets = _get_feat_target(c_v_feats, c_frame_masks)
    c_v_feats = _mask_img_feat(c_v_feats, c_frame_masks)

    batch['c_v_feats'] = c_v_feats  # already masked c_v_feats
    batch['c_v_masks'] = c_frame_masks
    batch['feat_targets'] = feat_targets  # the feature being masked
    
    # for meta info
    batch['vid_seg_ids'] = vid_seg_ids
    batch['verbs'] = verbs
    batch['vid_v_frames'] = vid_v_frames
    return batch

