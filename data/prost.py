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
import copy
import os
import re
import _pickle as cPickle

from transformers import RobertaTokenizerFast, AutoTokenizer
from .data import (pad_tensors, get_gather_index, _check_ngpu)
import datasets


def _get_txt_tgt_mask(txt_mask, n_frame):
	z = torch.zeros(n_frame, dtype=torch.bool)
	txt_mask_tgt = torch.cat([z, txt_mask], dim=0)
	return txt_mask_tgt


def get_label2token(example):
	label2token = {}
	label2token[0] = example['A']
	label2token[1] = example['B']
	label2token[2] = example['C']
	label2token[3] = example['D']
	return label2token


def _get_option_input_ids(example, tokenizer):
	# inspired by the prost official repository
	option_words = [example[opt] for opt in ['A', 'B', 'C', 'D']]
	tokenized_options = [tokenizer.tokenize(' ' + tok) for tok in option_words]
	option_input_ids = [tokenizer.convert_tokens_to_ids(tok)[0] for tok in tokenized_options]
	return option_input_ids


def load_prost(tokenizer, dataroot, max_frame_len, video_feat_dim, for_roberta=True, for_flava=True):
	ds = datasets.load_dataset('corypaik/prost', split='test')
	# inspired by the prost official repository
	
	if for_roberta:
		cache_path = os.path.join(dataroot, "prost", "cache", "roberta_txt")
		if not os.path.exists(cache_path):
			os.makedirs(os.path.join(dataroot, "prost/cache"), exist_ok=True)
			instances = []
			for example in ds:
				instance = {}
				# substitute [MASK] with <mask>
				# RoBERTa expects <mask> to be contain the space, e.g. `<mask>=' hi'`.
				example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token, example['question'])
				
				# format text input
				text_input = '{context} {question}'.format_map(example)
				
				# get text input
				tokens = tokenizer.tokenize(text_input)
				# add [CLS] token to the tokenized input
				input_id = tokenizer.convert_tokens_to_ids(['<s>']) + tokenizer.convert_tokens_to_ids(tokens)
				input_id = torch.tensor(input_id)
				
				mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
				mask_token_index = (input_id == mask_id).nonzero(as_tuple=True)[0]

				option_input_ids = _get_option_input_ids(example, tokenizer)
				
				instance['text_input'] = text_input
				instance['input_id'] = input_id
				instance['mask_token_index'] = mask_token_index
				instance['option_input_ids'] = torch.tensor(option_input_ids)
				
				# attention masks
				attn_masks = torch.ones(len(input_id), dtype=torch.long)
				instance['attn_masks'] = attn_masks

				# get label
				instance['label'] = example['label']
				# get label2token
				instance['label2token'] = get_label2token(example)
				
				# get meta information
				instance['group'] = example['group']
				
				# template name
				instance['name'] = example['name']
				
				instances.append(instance)
			
			cPickle.dump(instances, open(cache_path, "wb"))
		else:
			instances = cPickle.load(open(cache_path, "rb"))
			
	elif for_flava:
		cache_path = os.path.join(dataroot, "prost", "cache", "flava_txt")
		if not os.path.exists(cache_path):
			os.makedirs(os.path.join(dataroot, "prost/cache"), exist_ok=True)
			instances = []
			for example in ds:
				instance = {}
				# substitute [MASK] with <mask>
				example['question'] = re.sub(r'(\[MASK\])', tokenizer.mask_token, example['question'])
				
				# format text input
				text_input = '{context} {question}'.format_map(example)
				
				# get text input
				tokens = tokenizer.tokenize(text_input)
				input_id = tokenizer.convert_tokens_to_ids(['<s>']) + tokenizer.convert_tokens_to_ids(tokens)
				input_id = torch.tensor(input_id)
				
				mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
				mask_token_index = (input_id == mask_id).nonzero(as_tuple=True)[0]
				
				option_input_ids = _get_option_input_ids(example, tokenizer)
				
				instance['text_input'] = text_input
				instance['input_id'] = input_id
				instance['mask_token_index'] = mask_token_index
				instance['option_input_ids'] = torch.tensor(option_input_ids)
				
				# attention masks
				attn_masks = torch.ones(len(input_id), dtype=torch.long)
				instance['attn_masks'] = attn_masks
				
				# get label
				instance['label'] = example['label']
				# get label2token
				instance['label2token'] = get_label2token(example)
				
				# get meta information
				instance['group'] = example['group']
				
				# template name
				instance['name'] = example['name']
				
				instances.append(instance)
			
			cPickle.dump(instances, open(cache_path, "wb"))
		else:
			instances = cPickle.load(open(cache_path, "rb"))
	else:
		cache_path = os.path.join(dataroot,
		                          "prost",
		                          "cache",
		                          "roberta" + "_" + str(max_frame_len))
	
		if not os.path.exists(cache_path):
			os.makedirs(os.path.join(dataroot, "prost/cache"), exist_ok=True)
			instances = []
			for example in ds:
				instance = {}
				# substitute [MASK] with <mask>
				# RoBERTa expects <mask> to be contain the space, e.g. `<mask>=' hi'`.
				example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token, example['question'])
				
				# format text input
				text_input = '{context} {question}'.format_map(example)
				
				# get text input
				tokens = tokenizer.tokenize(text_input)
				# add [CLS] token to the tokenized input
				input_id = tokenizer.convert_tokens_to_ids(['<s>']) + tokenizer.convert_tokens_to_ids(tokens)
				mask_id = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
				txt_label = torch.tensor([0 if id == mask_id else -1 for id in input_id])
				
				instance['text_input'] = text_input
				instance['input_id'] = torch.tensor(input_id)
				instance['txt_label'] = txt_label
				
				option_input_ids = _get_option_input_ids(example, tokenizer)
				instance['option_input_ids'] = torch.tensor(option_input_ids)
		
				# get dummy video input
				instance['v_feat'] = torch.zeros(max_frame_len, video_feat_dim)
				nframes = max_frame_len
				
				# get dummy video frame ids
				div = nframes // 3
				clip_level_v_type_ids = [0] * div + [1] * div + [2] * div
				if len(clip_level_v_type_ids) < nframes:
					rest_num = nframes - len(clip_level_v_type_ids)
					clip_level_v_type_ids += [2] * rest_num
				clip_level_v_type_ids = torch.tensor(clip_level_v_type_ids).long()
				instance['clip_level_v_type_ids'] = clip_level_v_type_ids
			
				# attention masks
				# mask visual part for a language only task
				t_attn_masks = torch.ones(len(input_id), dtype=torch.long)
				v_attn_masks = torch.zeros(nframes, dtype=torch.long)
				attn_masks = torch.cat((t_attn_masks, v_attn_masks), 0)
				
				txt_mask_tgt = _get_txt_tgt_mask(txt_label != -1, nframes)
				instance['attn_masks'] = attn_masks
				instance['txt_mask_tgt'] = txt_mask_tgt
	
				# get label
				instance['label'] = example['label']
				# get label2token
				instance['label2token'] = get_label2token(example)
	
				# get meta information
				instance['group'] = example['group']
				
				# template name
				instance['name'] = example['name']
				
				instances.append(instance)
	
			cPickle.dump(instances, open(cache_path, "wb"))
		else:
			instances = cPickle.load(open(cache_path, "rb"))
	return instances
	
	
class ProstDataset(Dataset):
	def __init__(self, dataroot: str, max_frame_length: int = 20, video_feat_dim: int = 4352, for_roberta=False, for_flava=False):
		
		# Reformat + tokenize + cache
		self.for_roberta = for_roberta
		self.for_flava = for_flava

		tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
		if self.for_flava:
			# Flava's tokenizer is BERT tokenizer
			tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
		self._entries = load_prost(tokenizer, dataroot, max_frame_length, video_feat_dim, self.for_roberta, self.for_flava)
		
	def __len__(self):
		return len(self._entries)
	
	def __getitem__(self, i):
		output = []
		example = self._entries[i]
		
		if self.for_roberta or self.for_flava:
			output.append((example['text_input'],
						   example['input_id'],
			               example['mask_token_index'],
			               example['option_input_ids'],
			               example['attn_masks'],
			               example['label'],
			               example['label2token'],
			               example['group'],
			               example['name']))
		else:
			output.append((example['text_input'],
						   example['input_id'],
			               example['txt_label'],
			               example['option_input_ids'],
			               example['v_feat'],
			               example['clip_level_v_type_ids'],
			               example['attn_masks'],
			               example['txt_mask_tgt'],
			               example['label'],
			               example['label2token'],
			               example['group'],
			               example['name']))
		return output


def prost_collate_roberta(inputs):
	"""
	Return:
	:text_input   (n) clozed question
	:input_ids    (n, max_L) padded with 0
	:txt_labels   (n, max_L) padded with -1
	:option_input_ids (n, 4)
	:position_ids (n, max_L) padded with 0
	:attn_masks   (n, max_L) padded with 0
	"""
	
	(text_input, input_ids, mask_token_index, option_input_ids, attn_masks, labels, label2tokens, groups, names) = map(list, unzip(concat(inputs)))
	
	# text batches
	input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
	position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
	option_input_ids = pad_sequence(option_input_ids, batch_first=True, padding_value=-1)
	
	# attn masks
	attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

	batch = {'text_inputs': text_input,
	         'input_ids': input_ids,
	         'position_ids': position_ids,
	         'attn_masks': attn_masks,
	         'mask_token_indexes': mask_token_index,
	         'option_input_ids': option_input_ids,
	         'labels': labels,
	         'label2tokens': label2tokens,
	         'groups': groups,
	         'names': names}
	return batch


def prost_collate(inputs):
	"""
	Return:
	:text_input   (n) clozed question
	:input_ids    (n, max_L) padded with 0
	:txt_labels   (n, max_L) padded with -1
	:option_input_ids (n, 4)
	:position_ids (n, max_L) padded with 0
	:img_feat     (n, max_num_frame, feat_dim)
	:img_pos_ids  (n, max_num_frame)
	:img_type_ids     (n, max_num_frame) padded with 3
	:attn_masks   (n, max_{L + num_frames}) padded with 0
	"""

	(text_input, input_ids, txt_labels, option_input_ids, v_feats, clip_level_v_type_ids,
	 attn_masks, txt_masks, labels, label2tokens, groups, names) = map(list, unzip(concat(inputs)))
	
	# text batches
	txt_lens = [i.size(0) for i in input_ids]
	input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
	txt_mask_tgt = pad_sequence(txt_masks, batch_first=True, padding_value=0)
	txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
	position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
	option_input_ids = pad_sequence(option_input_ids, batch_first=True, padding_value=-1)

	# image batches
	num_fs = [f.size(0) for f in v_feats]
	v_feat = pad_tensors(v_feats, num_fs)
	
	# hard_coded padding value
	clip_level_v_type_ids = pad_sequence(clip_level_v_type_ids, batch_first=True, padding_value=3)
	
	clip_level_pos_ids = torch.arange(0, v_feat.size(1), dtype=torch.long).unsqueeze(0).expand(v_feat.size(0),
	                                                                                           -1).clone()
	
	# attn masks
	attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
	
	bs, max_vl, _ = v_feat.size()
	out_size = attn_masks.size(1)
	if max_vl > 0:
		gather_index = get_gather_index(txt_lens, num_fs, bs, max_vl, out_size)
	else:
		gather_index = None
		v_feat = None
	
	batch = {'text_inputs': text_input,
		     'input_ids': input_ids,
	         'position_ids': position_ids,
	         'v_feat': v_feat,
	         'c_v_type_ids': clip_level_v_type_ids,
	         'c_v_pos_ids': clip_level_pos_ids,
	         'attn_masks': attn_masks,
	         'gather_index': gather_index,
	         'txt_mask_tgt': txt_mask_tgt,
	         'txt_labels': txt_labels,
	         'option_input_ids': option_input_ids,
	         'labels': labels,
	         'label2tokens': label2tokens,
	         'groups': groups,
	         'names': names}
	return batch
