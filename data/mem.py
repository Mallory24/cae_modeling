"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pretrain MEM (Masked Effect Modeling) dataset
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd
import copy

from .data import VideoFeatSingleSubTokDataset, video_collate_clip_level, _check_ngpu


def _get_img_mask(mask_prob, num_frame, v_type_ids, mask_strategy="random"):
    # mask_strategy = ["random", "BEF-only", "ACT-only", "AFT-only"]
    assert num_frame == len(v_type_ids)
    if mask_strategy == "random":
        img_mask = [random.random() < mask_prob for _ in range(num_frame)]
    elif mask_strategy == "BEF-only":
        img_mask = [type_id == 0 for type_id in v_type_ids]
    elif mask_strategy == "ACT-only":
        img_mask = [type_id == 1 for type_id in v_type_ids]
    elif mask_strategy == "AFT-only":
        img_mask = [type_id == 2 for type_id in v_type_ids]
    
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_frame))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


def _create_v_type_ids(nframes):
    div = nframes // 3
    clip_level_v_type_ids = [0] * div + [1] * div + [2] * div
    if len(clip_level_v_type_ids) < nframes:
        rest_num = nframes - len(clip_level_v_type_ids)
        clip_level_v_type_ids += [2] * rest_num
    return clip_level_v_type_ids


class MemDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15, vid_type_size=4,
                 mask_strategy="random", video_frame_only=False, default_sampler=True):
        # mask_strategy = ["random", "BEF-only", "ACT-only", "AFT-only"]
        assert isinstance(vid_sub_db, VideoFeatSingleSubTokDataset)
        self.mask_prob = mask_prob
        self.vid_sub_db = vid_sub_db
        self.vid_type_size = vid_type_size - 1  # disregard [PAD]
        self.mask_strategy = mask_strategy
        self.video_frame_only = video_frame_only

        # only applicable for normal sampler
        if _check_ngpu() > 1 and default_sampler:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        # (input_id, c_v_feats, c_attn_masks, c_v_attn_masks) = self.vid_sub_db[vid]

        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]

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

        return (input_id, c_v_feats, clip_level_v_type_ids, attn_masks, c_v_attn_masks),\
               c_frame_mask, vid


def mem_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :pos_ids (n, max_L) padded with 0
    :img_feat     (n, max_num_frame, feat_dim)
    :img_pos_ids (n, max_num_frame) padded with 0
    :img_type_ids     (n, max_num_frame) padded with 3
    :img_masks  (n, max_num_frame) padded with 0
    :attn_masks   (n, max_{L + num_frames}) padded with 0
    :img_attn_masks   (n, max_num_frames) padded with 0
    :img_feat_target   (n, max_num_masked_frames) padded with 0
    """
    video_inputs, c_frame_masks, vids = map(list, unzip(inputs))
    # c_frame_masks: bz * num_frames of boolean, e.g., [True, False, False, False]
    batch = video_collate_clip_level(video_inputs)
    # batch [c_sub_input_ids] (bz, max_sl)
    # batch [c_sub_pos_ids] (bz, max_sl)
    # batch [c_v_feats] (bz, max_vl, k)
    # batch [clip_level_v_type_ids] (bz, max_vl)
    # batch [c_v_pos_ids] (bz, max_vl)
    # batch [c_attn_masks] (bz, max_vl+sl)
    # batch [c_v_attn_masks] (bz, max_vl) # for temporal modeling
    
    # mask features
    c_frame_masks = pad_sequence(c_frame_masks,
                                 batch_first=True, padding_value=0)
    c_v_feats = batch['c_v_feats']
    feat_targets = _get_feat_target(c_v_feats, c_frame_masks)
    c_v_feats = _mask_img_feat(c_v_feats, c_frame_masks)

    batch['c_v_feats'] = c_v_feats  # already masked c_v_feats
    batch['c_v_masks'] = c_frame_masks
    batch['feat_targets'] = feat_targets  # the feature being masked
    batch['vids'] = vids

    return batch
