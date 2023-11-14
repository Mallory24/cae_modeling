"""
Modified from HERO implementation
(https://github.com/linjieli222/HERO)

run evaluation of MAP
"""

import argparse
import os
from os.path import exists
from time import time
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pprint
from apex import amp
from horovod import torch as hvd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from model.model import VideoModelConfig

from data import (MAPFullEvalDataset, map_full_eval_collate, map_full_eval_collate_roberta, PrefetchLoader)
from load_data import (
	get_video_ids, load_video_sub_dataset,
	load_video_only_dataset)
from data.loader import move_to_cuda
from model.pretrain import HeroForPretraining

from utils.logger import LOGGER
from utils.const import VFEAT_DIM, VCMR_IOU_THDS
from utils.distributed import all_gather_list
from utils.misc import Struct, set_random_seed
from utils.basic_utils import (
	load_json, save_json)


def main(opts):
	hvd.init()
	n_gpu = hvd.size()
	device = torch.device("cuda", hvd.local_rank())
	torch.cuda.set_device(hvd.local_rank())
	rank = hvd.rank()
	LOGGER.info("device: {} n_gpu: {}, rank: {}, "
	            "16-bits training: {}".format(
		device, n_gpu, hvd.rank(), opts.fp16))
	
	if hvd.rank() != 0:
		LOGGER.disabled = True
	
	set_random_seed(opts.seed)
	torch.backends.cudnn.benchmark = False
	# torch.use_deterministic_algorithms(True)
	
	hps_file = f'{opts.output_dir}/log/hps.json'
	model_opts = Struct(load_json(hps_file))
	
	# load DBs and video dirs
	video_ids = json.load(open(f"{opts.sub_txt_db}/{opts.split}_ids.json", 'r'))
	
	if opts.task != "cae_video_only":
		video_db = load_video_sub_dataset(
			opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval,
			model_opts)
	else:
		video_db = None
		pass
	# TODO: TASK: given only the video clip, predict the action
	# txt_meta = load_json(
	#     os.path.join(opts.query_txt_db, "meta.json"))
	# video_db = load_video_only_dataset(
	#     opts.vfeat_db, txt_meta,
	#     model_opts.vfeat_interval,
	#     model_opts)
	# assert opts.split in opts.query_txt_db
	
	if opts.task != "cae_video_only":
		# given a video clip + the action-masked subtitle, predict the action token.
		inf_dataset = MAPFullEvalDataset
	else:
		# TODO: copy vr_video_only to map_video_only
		inf_dataset = None
		# inf_dataset = MAPVideoOnlyFullEvalDataset
		pass
	
	# Prepare model
	if opts.model_config == 'roberta':
		model = RobertaForMaskedLM.from_pretrained('roberta-base')
		
		eval_dataset = inf_dataset(video_ids,
		                           video_db,
		                           subtitle_only=opts.subtitle_only,
		                           distributed=n_gpu > 1,
		                           for_roberta=True)
		
		eval_dataloader = DataLoader(eval_dataset,
		                             batch_size=opts.batch_size,
		                             num_workers=opts.n_workers,
		                             pin_memory=opts.pin_mem,
		                             collate_fn=map_full_eval_collate_roberta,
		                             shuffle=False)
	
		eval_dataloader = PrefetchLoader(eval_dataloader)
	else:
		model_config = f'{opts.model_config}'  # initialize with the config of hero_finetune
		
		if exists(opts.checkpoint):
			ckpt_file = opts.checkpoint
		else:
			ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
		checkpoint = torch.load(ckpt_file)
		
		img_pos_embed_weight_key = (
				"v_encoder.f_encoder.img_embeddings" +
				".position_embeddings.weight")
		
		img_token_embed_weight_key = (
				"v_encoder.f_encoder.img_embeddings" +
				".token_type_embeddings.weight")
		
		assert img_pos_embed_weight_key in checkpoint
		assert img_token_embed_weight_key in checkpoint

		max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
	
		model = HeroForPretraining.from_pretrained(
			model_config,
			state_dict=checkpoint,
			vfeat_dim=VFEAT_DIM,
			max_frm_seq_len=max_frm_seq_len,
			lw_neg_ctx=model_opts.lw_neg_ctx,
			lw_neg_q=model_opts.lw_neg_q,
			ranking_loss_type=model_opts.ranking_loss_type,
			use_hard_negative=False,
			hard_pool_size=model_opts.hard_pool_size,
			margin=model_opts.margin,
			use_all_neg=model_opts.use_all_neg)
		
		eval_dataset = inf_dataset(video_ids,
		                           video_db,
		                           distributed=n_gpu > 1,
		                           subtitle_only=opts.subtitle_only,
		                           without_aft=opts.without_aft)

		eval_dataloader = DataLoader(eval_dataset,
		                             batch_size=opts.batch_size,
		                             num_workers=opts.n_workers,
		                             pin_memory=opts.pin_mem,
		                             collate_fn=map_full_eval_collate,
		                             shuffle=False)
		
		eval_dataloader = PrefetchLoader(eval_dataloader)
	
	model.to(device)
	if opts.fp16:
		model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
	
	st = time()
	is_roberta = True if opts.model_config == 'roberta' else False
	n_correct, n_word, results = validate_full_map(
		model, eval_dataloader, opts.split, opts, is_roberta, device)
	result_dir = f'{opts.output_dir}/MAP/{opts.subset}/results_{opts.split}'
	
	if not exists(result_dir) and rank == 0:
		os.makedirs(result_dir)
	
	n_correct = sum(all_gather_list(n_correct))
	n_word = sum(all_gather_list(n_word))
	tot_time = time() - st
	acc = n_correct / n_word
	
	LOGGER.info(f"evaluation finished in {int(tot_time)} seconds, "
	            f"acc: {acc * 100:.2f}")
	
	all_results = {}
	for id2res in all_gather_list(results):
		all_results.update(id2res)
	t = 'text_only' if opts.subtitle_only else 'text_visual'
	if hvd.rank() == 0:
		if opts.without_aft:
			save_json(
				all_results,
				f'{result_dir}/results_{opts.checkpoint}_{t}_no_aft_all.json')
		else:
			save_json(
				all_results,
				f'{result_dir}/results_{opts.checkpoint}_{t}_all.json')
		LOGGER.info('All results written......')


@torch.no_grad()
def validate_full_map(model, eval_loader, split, opts, is_roberta, device):
	LOGGER.info("start running full MAP evaluation"
	            f"on {opts.task} {split} split...")
	
	tokenizer = RobertaTokenizerFast.from_pretrained(opts.toker)
	
	model.eval()
	
	n_correct = 0
	n_word = 0
	results = {}
	for i, batch in enumerate(tqdm(eval_loader)):
		if is_roberta:
			# multiple mask tokens (subtokens of a verb) of one instance
			txt_mask_tgt = batch['txt_mask_tgt']
			outputs = model(input_ids=batch['input_ids'],
			                attention_mask=batch['attn_masks'],
			                position_ids=batch['position_ids'])
			logits = outputs.logits
			txt_mask_tgt = txt_mask_tgt.unsqueeze(-1).expand_as(logits)
			# print("txt mask tgt", txt_mask_tgt.unsqueeze(-1))
			# print("txt mask tgt size:", txt_mask_tgt.unsqueeze(-1).size())

			scores = logits[txt_mask_tgt].contiguous().view(-1, logits.size(-1))
		
			# print("scores", scores)
			# print("scores size:", scores.size())
		
			# mask_token_indexes = torch.as_tensor(mask_token_indexes).to(device)
			# scores = torch.index_select(outputs.logits, dim=1, index=mask_token_indexes)
		# reuse MAM head to perform inference
		else:
			scores = model(batch, task='mam', compute_loss=False)
			
		labels = batch['txt_labels']
		n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
		n_word += labels.numel()
		
		answers = [i for i in scores.max(
			dim=-1, keepdim=False)[1].cpu().tolist()]
		labels = [i for i in batch['txt_labels'].cpu().tolist()]
		
		vid_seg_ids = batch['vid_seg_ids']
		ground_truth_verbs = batch['verbs']

		for vid_seg_id, ground_truth_verb, answer, label in zip(vid_seg_ids, ground_truth_verbs, answers, labels):
			results[vid_seg_id] = {}
			results[vid_seg_id]['label'] = ground_truth_verb
			if tokenizer.convert_ids_to_tokens(answer) is not None:
				results[vid_seg_id]['prediction'] = tokenizer.convert_ids_to_tokens(answer).replace('Ġ', '')
				# print("answer", answer)
				# print("decoded answer", tokenizer.convert_ids_to_tokens(answer))
				# results[vid_seg_id]['prediction'] = ''
				# for a in answer:
				# 	results[vid_seg_id]['prediction'] += tokenizer.convert_ids_to_tokens(answer).replace('Ġ', '')
			else:
				results[vid_seg_id]['prediction'] = 'Not decoded'
			
			# all subtokens of a verb token have to be correct to be considered as correct
			results[vid_seg_id]['correct'] = answer == label
	return n_correct, n_word, results


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# Required parameters
	parser.add_argument("--sub_txt_db",
	                    default="/txt_db/exp_10_per",
	                    type=str,
	                    help="The input video subtitle corpus. (LMDB)")
	parser.add_argument("--vfeat_db",
	                    default="/video_db/exp_10_per", type=str,
	                    help="The input video frame features.")
	parser.add_argument('--toker', default='roberta-base',
	                    help='which RoBerTa tokenizer to be used')
	parser.add_argument("--split", choices=["train", "val", "test"],
	                    default="val", type=str,
	                    help="The input evaluation split")
	parser.add_argument("--subset", default="10_per", type=str,
	                    help="The subset")
	parser.add_argument("--task", choices=["cae_video_sub",
	                                       "cae_video_only"],
	                    default="cae_video_sub", type=str,
	                    help="The evaluation map task")
	parser.add_argument('--subtitle_only',
	                    action='store_true',
	                    help="Whether to only have textual input present")
	parser.add_argument('--without_aft',
	                    action='store_true',
	                    help="Whether to leave out [AFT] frames in the visual input")
	parser.add_argument("--checkpoint",
	                    default=None, type=str,
	                    help="pretrained model checkpoint steps")
	parser.add_argument("--model_config",
	                    default=None, type=str,
	                    help="pretrained model config file")
	parser.add_argument("--batch_size",
	                    default=64, type=int,
	                    help="number of video segments in a batch")
	parser.add_argument(
		"--output_dir", default=None, type=str,
		help="The output directory where the model checkpoints will be "
		     "written.")

	# device parameters
	parser.add_argument('--fp16',
	                    action='store_true',
	                    help="Whether to use 16-bit float precision instead "
	                         "of 32-bit")
	parser.add_argument('--n_workers', type=int, default=0,
	                    help="number of data workers")
	parser.add_argument('--seed', type=int, default=77,
	                    help="seed number")
	parser.add_argument('--pin_mem', action='store_true',
	                    help="pin memory")
	
	args = parser.parse_args()
	
	main(args)
