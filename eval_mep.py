"""
Modified from HERO implementation
(https://github.com/linjieli222/HERO)

run evaluation of MEP
"""

import argparse
import os
import random
from os.path import exists
from time import time
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pprint
from apex import amp
from horovod import torch as hvd
from transformers import RobertaTokenizerFast

from data import (MEPFullEvalDataset, mep_full_eval_collate, PrefetchLoader, MemNegativeSampler)
from load_data import (get_video_ids, load_video_sub_dataset, load_video_only_dataset)
from data.loader import move_to_cuda
from model.pretrain import HeroForPretraining

from utils.logger import LOGGER
from utils.const import VFEAT_DIM, VCMR_IOU_THDS
from utils.distributed import all_gather_list
from utils.misc import Struct, set_random_seed
from utils.basic_utils import (load_json, save_json)

from collections import defaultdict


def get_average_acc_pre_instance(results):
	results_avg = {}
	for vid_seg_id in results.keys():
		results_avg[vid_seg_id] = {'hard_correct': 0}
		# results_avg[vid_seg_id] = {'hard_correct': 0, 'soft_correct': 0}
		if len(results[vid_seg_id]['hard_correct']) > 1:    # for a video segment that contains multiple nouns
			print("vid_seg_id", vid_seg_id)
		results_avg[vid_seg_id]['hard_correct'] = sum(results[vid_seg_id]['hard_correct'])/len(results[vid_seg_id]['hard_correct'])
		# results_avg[vid_seg_id]['soft_correct'] = sum(results[vid_seg_id]['soft_correct'])/len(results[vid_seg_id]['soft_correct'])
	return results_avg
		
		
def get_candidate_video_frame_idx(vid_seg_ids, c_v_masks):
	candidate_vid_frame_idx = []
	# prepare the masked ones
	for vid_idx, masked_frames in enumerate(c_v_masks):
		for idx, masked_frame in enumerate(masked_frames):
			if masked_frame == True:
				vid_seg_frame_idx = str(vid_seg_ids[vid_idx]) + "_" + str(idx)
				candidate_vid_frame_idx.append(vid_seg_frame_idx)
	
	# prepare the unmasked ones
	for vid_idx, masked_frames in enumerate(c_v_masks):
		for idx, masked_frame in enumerate(masked_frames):
			if masked_frame == False:
				vid_seg_frame_idx = str(vid_seg_ids[vid_idx]) + "_" + str(idx)
				candidate_vid_frame_idx.append(vid_seg_frame_idx)
	
	return candidate_vid_frame_idx


def _create_v_type_ids(nframes):
    div = nframes // 3
    clip_level_v_type_ids = [0] * div + [1] * div + [2] * div
    if len(clip_level_v_type_ids) < nframes:
        rest_num = nframes - len(clip_level_v_type_ids)
        clip_level_v_type_ids += [2] * rest_num
    return clip_level_v_type_ids


def get_candidates(vid_seg_lst, vid2nframe):
	"""
	input: [vid_seg_idA, vid_seg_idB]
	output: [vid_seg_idA-0-0, vid_seg_idA-1-0, vid_seg_idA-2-0,
	vid_seg_idB-0-0, vid_seg_idB-1-0, vid_seg_idB-2-0]
	"""
	# {[BEF]: 0, [ACT]: 1, [AFT]: 2, [PAD]: 3}
	vid_seg_id_nframes = {vid_seg_id: vid2nframe.get(vid_seg_id) for vid_seg_id in vid_seg_lst}
	vid_seg_id_nframes_types = {k: _create_v_type_ids(v) for k, v in vid_seg_id_nframes.items()}
	
	candidates = []
	for k, nframes_types in vid_seg_id_nframes_types.items():
		vid_seg_nframes_types = [k + '-' + str(v) for v in nframes_types]
		# [vid_seg_idA-0, vid_seg_idA-1, vid_seg_idA-2]
		# print('vid_seg_nframes_types', vid_seg_nframes_types)
		
		count = defaultdict(int)
		vid_seg_nframes_types_with_index = []
		
		# intended outcome: [vid_seg_idA-0-0, vid_seg_idA-1-0, vid_seg_idA-2-0]
		for t in vid_seg_nframes_types:
			vid_seg_nframes_types_with_index.append(str(t)+'-'+str(count[t]))
			count[t] += 1
		
		candidates.extend(vid_seg_nframes_types_with_index)
	# print('candidates', candidates)
	return candidates


def get_candidates_per_instance(batch_neg, vid_seg_id):
	# get the candidates per video clip instance
	"""
	input:
	batch_neg = [vid_seg_idA-0-0, vid_seg_idA-1-0, vid_seg_idA-2-0,
	vid_seg_idB-0-0, vid_seg_idB-1-0, vid_seg_idB-2-0],
	vid_seg_id: vid_seg_idA
	output:
	(remove the [AFT] frame) [vid_seg_idA-0-0, vid_seg_idA-1-0,
	vid_seg_idB-0-0, vid_seg_idB-1-0, vid_seg_idB-2-0]
	"""
	remove = vid_seg_id + '-' + str(2)
	candidates = [neg for neg in batch_neg if not neg.startswith(remove)]
	return candidates


def get_labels_per_instance(batch_neg, vid_seg_id):
	# get the labels per video clip instance
	"""
	input:
	batch_neg = [vid_seg_idA-0-0, vid_seg_idA-1-0, vid_seg_idA-2-0,
	vid_seg_idB-0-0, vid_seg_idB-1-0, vid_seg_idB-2-0],
	vid_seg_id: vid_seg_idA
	output: vid_seg_idA-2-0]
	"""
	aft = vid_seg_id + '-' + str(2)
	labels = [neg for neg in batch_neg if neg.startswith(aft)]
	return sorted(labels)
	
	
def validate_full_mep_with_random_baseline(neg, vid2nframe, batch_size, opts):
	LOGGER.info("start running full MEP evaluation"
	            f"on {opts.task} {opts.split} split...")
	# Quick approach: iterate through the negative file
	# TO THINK: how to perform in a batch manner <- use eval_loader
	# TODO: Another Baseline: Use the visual feature as it is to compute similarity
	results = {}
	n_neg = 0
	n_feat = 0
	# Get all the candidates (intra_video_clip [AFT] is included) at the video frame level:
	batch_candidates = []
	batch_vid_seg_ids = []
	for vid in neg.keys():
		vid_seg_lst = neg[vid]
		while len(vid_seg_lst) >= batch_size:
			selected_vid_seg_lst = vid_seg_lst[:batch_size]
			batch_vid_seg_ids.append(selected_vid_seg_lst)
			candidates = get_candidates(selected_vid_seg_lst, vid2nframe)
			batch_candidates.append(candidates)
			vid_seg_lst = vid_seg_lst[batch_size:]
			
		if len(vid_seg_lst) > 1:
			selected_vid_seg_lst = vid_seg_lst
			batch_vid_seg_ids.append(selected_vid_seg_lst)
			candidates = get_candidates(vid_seg_lst, vid2nframe)
			batch_candidates.append(candidates)
		
	assert len(batch_vid_seg_ids) == len(batch_candidates)
	print('Num of batch:', len(batch_candidates))
	
	# Iterate through the batch, perform random selection for each video clip inside the batch
	# on each of its [AFT] frame (note that the candidate should not contain intra-video-clip [AFT] frame type)
	# TODO: enhance time efficiency
	for batch, batch_neg in zip(batch_vid_seg_ids, batch_candidates):
		for vid_seg_id in batch:
			# get pre video segment labels
			AFT_labels = get_labels_per_instance(batch_neg, vid_seg_id)
			# print('vid_seg_id', vid_seg_id)
			per_vid_seg_candidates = get_candidates_per_instance(batch_neg, vid_seg_id)
			# Double check if the gt AFT label is in the candidate, but not the intra-video-clip
			n_feat += len(AFT_labels)
			n_neg += len(per_vid_seg_candidates)
			
			# randomly choose num_[AFT] times
			select = []
			for aft_label in AFT_labels:
				per_vid_seg_candidates_with_gt = per_vid_seg_candidates + [aft_label]
				# print(per_vid_seg_candidates_with_gt)
				select.append(random.choice(per_vid_seg_candidates_with_gt))
			# print('Select', select)
			
			if vid_seg_id not in results:
				results[vid_seg_id] = {}
				# soft_correct = False
				# if len(set(AFT_labels) & set(select)) == len(AFT_labels):
				# 	soft_correct = True
				results[vid_seg_id]['ground_truth_video_frame'] = AFT_labels
				results[vid_seg_id]['predicted_video_frame'] = select
				# results[vid_seg_id]['soft_correct'] = [soft_correct]
				results[vid_seg_id]['hard_correct'] = [AFT_labels == select]
	return n_feat, n_neg, results
	
	
@torch.no_grad()
def validate_full_mep(model, eval_loader, split, opts):
	LOGGER.info("start running full MEP evaluation"
	            f"on {opts.task} {split} split...")
	
	model.eval()
	
	cosine = 0
	n_feat = 0
	n_neg = 0
	results = {}
	for i, batch in enumerate(tqdm(eval_loader)):
		feats, neg_feats = model(batch, task='mem-nce', compute_loss=False)
		pos_feats = batch['feat_targets']
		c_v_masks = batch['c_v_masks']
		
		logits = model.v_encoder.mem_nce(feats, pos_feats, neg_feats, compute_loss=False, c_v_masks=c_v_masks)
		# print("logits", logits)
		# print("masked_feats size", feats.size())
		# print("neg_feats size", neg_feats.size())
		
		targets = torch.arange(0, logits.size(0), dtype=torch.long, device=logits.device)
		# print("targets", targets)
		
		# val_loss += F.cross_entropy(logits, targets, reduction='sum').item()
		# val_l2 += F.mse_loss(feats, pos_feats, reduction='none'
		#                      ).sum(dim=1).sqrt().sum().item()
		# n_correct += (logits.max(dim=-1)[1] == targets).sum().item()
		cosine += F.cosine_similarity(feats, pos_feats, dim=-1).sum().item()
		# print("cosine", cosine)
		nf = pos_feats.size(0)
		n_feat += nf
		# TO CHECK: why neg_feats.size(0) * nf
		n_neg += neg_feats.size(0) * nf
		
		# answers are aggregated over a batch
		answers = [i for i in logits.max(dim=-1, keepdim=False)[1].cpu().tolist()]
		# print("answers", answers)
		labels = [i for i in targets.cpu().tolist()]
		# print("labels", labels)
		
		vid_seg_ids = batch['vid_seg_ids']
		ground_truth_verbs = batch['verbs']
		candidate_vid_frame_idx = get_candidate_video_frame_idx(vid_seg_ids, c_v_masks)
		
		start_idx = 0
		
		for vid_seg_id, ground_truth_verb, c_v_mask in zip(vid_seg_ids, ground_truth_verbs, c_v_masks):
			if vid_seg_id not in results:
				# this is for object_based negative sampling since one video clip can have multiple objects
				results[vid_seg_id] = {'verb': None,
				                       'ground_truth_video_frame': [],
				                       'predicted_video_frame': [],
				                       'hard_correct': [],
				                       'soft_correct': [],
				                       'competitors': vid_seg_ids}
			
			results[vid_seg_id]['verb'] = ground_truth_verb
			# print('c_v_mask', c_v_mask)
			num_masked_frames = len(c_v_mask.nonzero())
			# print('num_masked_frames', num_masked_frames)
			
			end_idx = start_idx + num_masked_frames
			# print('end_idx', end_idx)
			per_instance_labels = labels[start_idx:end_idx]
			# print('label', per_instance_labels)
			groud_truth_values = [candidate_vid_frame_idx[label] for label in per_instance_labels]
			per_instance_answers = answers[start_idx:end_idx]
			# print('answer', per_instance_answers)
			predicted_values = [candidate_vid_frame_idx[answer] for answer in per_instance_answers]
			
			results[vid_seg_id]['ground_truth_video_frame'] = groud_truth_values  # (vid_seg_ids_frame_idx)
			# print('ground-truth', results[vid_seg_id]['groud_truth_video_frame'])
			results[vid_seg_id]['predicted_video_frame'] = predicted_values  # (vid_seg_ids_frame_idx)
			# print('predicted', results[vid_seg_id]['predicted_video_frame'])
			
			# It is considered correct only when the [AFT] video frame idxes are correctly ordered.
			hard_correct = per_instance_labels == per_instance_answers
			results[vid_seg_id]['hard_correct'].append(hard_correct)
			
			# Not relevent anymore:
			# It is considered correct as long as the predicted video frame idxes fall into [AFT] category,
			# regardless of the order.
			# TO FIX?
			# if len(set(per_instance_answers) & set(per_instance_labels)) == len(per_instance_answers):
			# 	soft_correct = True
			# soft_correct = set(per_instance_answers).isdisjoint(set(per_instance_labels)) is not True
			# results[vid_seg_id]['soft_correct'].append(soft_correct)

			start_idx += num_masked_frames  # print('start_idx', start_idx)
			
	return cosine, n_feat, n_neg, results


def main(opts):
	hvd.init()
	n_gpu = hvd.size()
	device = torch.device("cuda", hvd.local_rank())
	torch.cuda.set_device(hvd.local_rank())
	rank = hvd.rank()
	LOGGER.info("device: {} n_gpu: {}, rank: {}, "
	            "16-bits training: {}".format(device, n_gpu, hvd.rank(), opts.fp16))
	
	if hvd.rank() != 0:
		LOGGER.disabled = True
	
	set_random_seed(opts.seed)
	torch.backends.cudnn.benchmark = False
	
	hps_file = f'{opts.output_dir}/log/hps.json'
	model_opts = Struct(load_json(hps_file))
	model_config = f'{opts.model_config}'  # initialize with the config of hero_finetune
	
	# load DBs and video dirs
	video_ids = json.load(open(f"{opts.sub_txt_db}/{opts.split}_ids.json", 'r'))
	
	if opts.task != "cae_video_only":
		video_db = load_video_sub_dataset(opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval, model_opts)
	else:
		video_db = None
		pass
	# given only video clip, predict the post condition in the visual space.
	# txt_meta = load_json(  #     os.path.join(opts.query_txt_db, "meta.json"))
	# video_db = load_video_only_dataset(  #     opts.vfeat_db, txt_meta,  #     model_opts.vfeat_interval,  #     model_opts)  # assert opts.split in opts.query_txt_db
	
	if opts.task != "cae_video_only":
		inf_dataset = MEPFullEvalDataset
	else:
		# TODO: copy vr_video_only to mep_video_only
		inf_dataset = None
		# inf_dataset = MEPVideoOnlyFullEvalDataset
		pass
	
	if opts.negative_sampling_strategy == "randomized":
		eval_dataset = inf_dataset(video_ids, video_db, video_frame_only=opts.video_frame_only, distributed=n_gpu > 1)
		eval_dataloader = DataLoader(eval_dataset, batch_size=opts.batch_size, num_workers=opts.n_workers,
		                             pin_memory=opts.pin_mem, collate_fn=mep_full_eval_collate, shuffle=False)
		eval_dataloader = PrefetchLoader(eval_dataloader)
	
	elif opts.negative_sampling_strategy in ["object-based", "video-based"]:
		eval_dataset = inf_dataset(video_ids, video_db, video_frame_only=opts.video_frame_only, distributed=n_gpu > 1,
		                           default_sampler=False)
		neg = json.load(open(f"{opts.sub_txt_db}/{opts.negative_examples}"))
		sampler = MemNegativeSampler(neg, video_ids, opts.batch_size, opts.split)
		eval_dataloader = DataLoader(eval_dataset, batch_sampler=sampler, num_workers=opts.n_workers,
		                             pin_memory=opts.pin_mem, collate_fn=mep_full_eval_collate)
		eval_dataloader = PrefetchLoader(eval_dataloader)
	
	# Prepare model
	st = time()
	if opts.model_config == 'baseline':
		neg = json.load(open(f"{opts.sub_txt_db}/{opts.negative_examples}"))
		vid2nframe = json.load(open(f"{opts.vfeat_db}/id2nframe.json"))
		cosine = 1  # default value
		n_feat = 1  # default value
		# n_neg = 1   # default value
		n_feat, n_neg, results = validate_full_mep_with_random_baseline(neg, vid2nframe, opts.batch_size, opts)
	else:
		if exists(opts.checkpoint):
			ckpt_file = opts.checkpoint
		else:
			ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
		checkpoint = torch.load(ckpt_file)
		
		img_pos_embed_weight_key = ("v_encoder.f_encoder.img_embeddings" + ".position_embeddings.weight")
		img_token_embed_weight_key = ("v_encoder.f_encoder.img_embeddings" + ".token_type_embeddings.weight")
		
		assert img_pos_embed_weight_key in checkpoint
		assert img_token_embed_weight_key in checkpoint
		
		max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
		
		model = HeroForPretraining.from_pretrained(model_config, state_dict=checkpoint, vfeat_dim=VFEAT_DIM,
		                                           max_frm_seq_len=max_frm_seq_len, lw_neg_ctx=model_opts.lw_neg_ctx,
		                                           lw_neg_q=model_opts.lw_neg_q,
		                                           ranking_loss_type=model_opts.ranking_loss_type, use_hard_negative=False,
		                                           hard_pool_size=model_opts.hard_pool_size, margin=model_opts.margin,
		                                           use_all_neg=model_opts.use_all_neg)
		
		model.to(device)
		if opts.fp16:
			model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
		
		cosine, n_feat, n_neg, results = validate_full_mep(model, eval_dataloader, opts.split, opts)
	
	results_avg = get_average_acc_pre_instance(results)
	result_dir = f'{opts.output_dir}/MEP/results_{opts.split}/{opts.negative_sampling_strategy}/{opts.subset}'
	
	if not exists(result_dir) and rank == 0:
		os.makedirs(result_dir)
	
	cosine = sum(all_gather_list(cosine))
	n_feat = sum(all_gather_list(n_feat))
	n_neg = sum(all_gather_list(n_neg))
	
	tot_time = time() - st
	
	val_hard_acc = sum([results_avg[vid_seg_id]['hard_correct'] for vid_seg_id in results_avg.keys()]) / len(
		results_avg)
	# val_soft_acc = sum([results_avg[vid_seg_id]['soft_correct'] for vid_seg_id in results_avg.keys()]) / len(
	# 	results_avg)
	
	cosine = cosine / n_feat
	
	LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
	            f"hard acc: {val_hard_acc * 100:.2f}, "
	            # f"soft acc: {val_soft_acc * 100:.2f},"
	            f"cosine: {cosine:.2f}, "
	            f"(average {n_neg / n_feat:.0f} negatives)")
	
	all_results = {}
	for id2res in all_gather_list(results):
		all_results.update(id2res)
	
	v = 'visual_only' if opts.video_frame_only else 'text_visual'
	if hvd.rank() == 0:
		save_json(all_results, f'{result_dir}/results_{opts.checkpoint}_{v}_all.json')
		LOGGER.info('All results written......')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# Required parameters
	parser.add_argument("--sub_txt_db", default="/txt_db/exp_10_per", type=str,
	                    help="The input video subtitle corpus. (LMDB)")
	parser.add_argument("--vfeat_db", default="/video_db/exp_10_per", type=str, help="The input video frame features.")
	parser.add_argument('--toker', default='roberta-base', help='which RoBerTa tokenizer to be used')
	parser.add_argument("--split", choices=["train", "val", "test"], default="val", type=str,
	                    help="The input evluation split")
	parser.add_argument("--subset", default="exp_10_per", type=str,
	                    help="The dataset size")
	parser.add_argument("--task", choices=["cae_video_sub", "cae_video_only"], default="cae_video_sub", type=str,
	                    help="The evaluation map task")
	parser.add_argument('--video_frame_only',
	                    action='store_true',
	                    help="Whether to only have visual input present")
	parser.add_argument("--negative_sampling_strategy", choices=["randomized", "video-based", "object-based"],
	                    default="randomized", type=str, help="The negative sampling strategy of the evaluation")
	parser.add_argument("--negative_examples", default="neg/val_object_based_negatives.json", type=str,
	                    help="The negative examples")
	parser.add_argument("--checkpoint", default=None, type=str, help="pretrained model checkpoint steps")
	parser.add_argument("--model_config", default=None, type=str, help="pretrained model config file")
	parser.add_argument("--batch_size", default=64, type=int, help="number of video segments in a batch")
	parser.add_argument("--output_dir", default=None, type=str,
		help="The output directory where the model checkpoints will be "
		     "written.")
	
	# device parameters
	parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead "
	                                                        "of 32-bit")
	parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
	parser.add_argument('--seed', type=int, default=77, help="seed number")
	parser.add_argument('--pin_mem', action='store_true', help="pin memory")
	
	args = parser.parse_args()
	
	main(args)
