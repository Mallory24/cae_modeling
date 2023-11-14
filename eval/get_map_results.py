"""
Get evaluation results for MAP
"""

import argparse
import json
import os
import pandas as pd
from collections import Counter
from pprint import pprint
from utils.basic_utils import (
	load_json, save_json, save_lines, read_lines)
import shutil
import scipy.stats


def get_seen_unseen_vid_segs(result_vid_segs: set, eval_table: dict, data_split: str):
	seen_vid_segs = set()
	unseen_vid_segs = set()
	
	seen_verb_vid_segs = {}
	unseen_verb_vid_segs = {}
	
	seen_verbs = set()
	unseen_verbs = set()
	for F in eval_table.keys():
		for verb_cls in eval_table[F].keys():
			if verb_cls == 'seen classes':
				seen_verbs.update(set(eval_table[F][verb_cls].keys()))
			elif verb_cls == 'unseen classes':
				unseen_verbs.update(set(eval_table[F][verb_cls].keys()))
			for v in eval_table[F][verb_cls]:
				for split in eval_table[F][verb_cls][v].keys():
					if split == data_split:
						for d in eval_table[F][verb_cls][v][split].keys():
							vid_segs = eval_table[F][verb_cls][v][split][d]
							if verb_cls == 'seen classes':
								# some does not have video feature
								seen_vid_segs |= set(vid_segs).intersection(result_vid_segs)
								if v not in seen_verb_vid_segs:
									seen_verb_vid_segs[v] = set(vid_segs).intersection(result_vid_segs)
								seen_verb_vid_segs[v] |= set(vid_segs).intersection(result_vid_segs)
							elif verb_cls == 'unseen classes':
								unseen_vid_segs |= set(vid_segs).intersection(result_vid_segs)
								if v not in unseen_verb_vid_segs:
									unseen_verb_vid_segs[v] = set(vid_segs).intersection(result_vid_segs)
								unseen_verb_vid_segs[v] |= set(vid_segs).intersection(result_vid_segs)
	
	return list(seen_vid_segs), list(unseen_vid_segs), seen_verb_vid_segs, unseen_verb_vid_segs, seen_verbs, unseen_verbs


def get_frames_vid_segs(result_vid_segs: set, eval_table: dict, data_split: str):
	
	FN_frames_instances = {F: {'seen_verb': {}, 'unseen_verb': {}} for F in eval_table.keys()}
	# divided by verb classes
	
	for F in eval_table.keys():
		for verb_cls in eval_table[F].keys():
			for v in eval_table[F][verb_cls]:
				for split in eval_table[F][verb_cls][v].keys():
					if split == data_split:
						for d in eval_table[F][verb_cls][v][split].keys():
							vid_segs = eval_table[F][verb_cls][v][split][d]
							if verb_cls == 'seen classes':
								if v not in FN_frames_instances[F]['seen_verb']:
									FN_frames_instances[F]['seen_verb'][v] = set(vid_segs).intersection(result_vid_segs)
								FN_frames_instances[F]['seen_verb'][v] |= set(vid_segs).intersection(result_vid_segs)
							elif verb_cls == 'unseen classes':
								if v not in FN_frames_instances[F]['unseen_verb']:
									FN_frames_instances[F]['unseen_verb'][v] = set(vid_segs).intersection(result_vid_segs)
								FN_frames_instances[F]['unseen_verb'][v] |= set(vid_segs).intersection(result_vid_segs)
	
	return FN_frames_instances

# This function is for quick testing
# TODO: write a function to see how many seen verb classes/unseen verb classes per Frame
# def get_frames_by_split(eval_table):
# 	FN_frames_instances = {F: {'train': 0, 'val': 0, 'test': 0} for F in eval_table.keys()}
#
# 	for F in eval_table.keys():
# 		for verb_cls in eval_table[F].keys():
# 			for v in eval_table[F][verb_cls]:
# 				for split in eval_table[F][verb_cls][v].keys():
# 					for d in eval_table[F][verb_cls][v][split].keys():
# 						vid_seg_counts = len(eval_table[F][verb_cls][v][split][d])
# 						FN_frames_instances[F][split] += vid_seg_counts
#
# 	not_in_train = 0
# 	for F in FN_frames_instances.keys():
# 		if FN_frames_instances[F]['train'] == 0:
# 			not_in_train += 0
# 			# print(F)
# 	# print(not_in_train)


def get_wrong_vid_seg(results: dict):
	df = pd.DataFrame.from_dict(results).T
	incorrect_df = df[df["correct"] == True]
	wrong_vid_segs = incorrect_df.index.values.tolist()
	return wrong_vid_segs


def get_meta_info(eval_table: dict):
	FN2verbs = {F: {'seen_verbs': set(), 'unseen_verbs': set()} for F in eval_table.keys()}
	# divided by verb classes
	seen_verb_classes = set()
	unseen_verb_classes = set()
	seen_verb_train_instances = {}
	seen_verb_test_instances = {}
	unseen_verb_train_instances = {}
	unseen_verb_test_instances = {}

	for F in eval_table.keys():
		seen_verb_classes.update(eval_table[F]['seen classes'].keys())
		unseen_verb_classes.update(eval_table[F]['unseen classes'].keys())
		
		FN2verbs[F]['seen_verbs'].update(eval_table[F]['seen classes'].keys())
		FN2verbs[F]['unseen_verbs'].update(eval_table[F]['unseen classes'].keys())
		
		for type in eval_table[F].keys():
			for verb in eval_table[F][type].keys():
				total_train_instances = sum([len(v) for v in eval_table[F][type][verb]['train'].values()])
				total_test_instances = sum([len(v) for v in eval_table[F][type][verb]['test'].values()])

				if type == "seen classes":
					seen_verb_train_instances[verb] = total_train_instances
					seen_verb_test_instances[verb] = total_test_instances
				else:
					unseen_verb_train_instances[verb] = total_train_instances
					unseen_verb_test_instances[verb] = total_test_instances
	
	return FN2verbs, seen_verb_classes, unseen_verb_classes, seen_verb_train_instances, unseen_verb_train_instances, seen_verb_test_instances, unseen_verb_test_instances


def get_harmonic_mean(seen_acc, unseen_acc):
	if seen_acc == 0 or unseen_acc == 0:
		return ''
	else:
		return round(2*(seen_acc * unseen_acc)/(seen_acc + unseen_acc), 4)


def get_verb_num_instances(verb_classes, results):
	df = pd.DataFrame.from_dict(results).T
	verb_instances = {}
	for verb in verb_classes:
		verb_df = df[df["label"] == verb]
		verb_instances[verb] = len(verb_df)
	return verb_instances


def calculate_macro_acc(verb_classes, results):
	df = pd.DataFrame.from_dict(results).T
	all_verb_acc = {}
	for verb in verb_classes:
		verb_df = df[df["label"] == verb]
		verb_corect_df = verb_df[verb_df["correct"] == True]
		if len(verb_df) != 0:
			all_verb_acc[verb] = len(verb_corect_df)/len(verb_df)
		
	all_acc = list(all_verb_acc.values())
	avg_acc = round(sum(all_acc)/len(all_acc), 4)
	return avg_acc, all_verb_acc


def calculate_micro_acc(results):
	df = pd.DataFrame.from_dict(results).T
	correct_df = df[df["correct"] == True]
	incorrect_df = df[df["correct"] == False]
	assert len(correct_df) + len(incorrect_df) == len(df)
	return round(len(correct_df)/len(df), 4)


def get_FN_stats(FN2verbs, all_verb_acc):
	FN_frame_accuracies = {F: {'seen_verbs': FN2verbs[F]['seen_verbs'],
	                           'unseen_verbs': FN2verbs[F]['unseen_verbs']}
	                       for F in FN2verbs.keys()}
	
	for F in FN_frame_accuracies:
		for type in FN2verbs[F].keys():
			acc = [all_verb_acc.get(v) for v in FN_frame_accuracies[F][type]]
			if len(acc) > 0:
				avg_seen_acc = round(sum(acc) / len(acc), 4)
				FN_frame_accuracies[F][f'{type}_acc'] = avg_seen_acc
			else:
				FN_frame_accuracies[F][f'{type}_acc'] = 0

		FN_frame_accuracies[F]['harmonic_acc'] = get_harmonic_mean(FN_frame_accuracies[F]['seen_verbs_acc'],
		                                                           FN_frame_accuracies[F]['unseen_verbs_acc'])
	return FN_frame_accuracies


def output_df(indexes, columns, dict):
	df = pd.DataFrame(index=indexes, columns=columns)
	for idx in indexes:
		for c in columns:
			df.loc[idx, c] = dict.get(idx)[c]
	return df


def main(opts):
	acc_type = ['seen_verb_acc', 'unseen_verb_acc', 'harmonic_acc', 'micro_acc']
	
	models = opts.models.split(',')
	ckpts = opts.checkpoints.split(',')
	
	model_ckpts = [model + '_' + ckpt for model, ckpt in zip(models, ckpts)]
	
	df = pd.DataFrame(index=acc_type, columns=model_ckpts)

	# result_vid_segs = load_json(opts.video_segment_ids)
	eval_table = load_json(opts.eval_table)
	# raw_data = load_json(opts.raw_data)
	
	# TODO: consider remove
	# seen_vid_segs, unseen_vid_segs, seen_verb_vid_segs, unseen_verb_vid_segs, seen_verbs, unseen_verbs = get_seen_unseen_vid_segs(result_vid_segs, eval_table, opts.split)
	# print('seen verb total datapoints:', len(seen_vid_segs))
	# print('unseen verb total datapoints:', len(unseen_vid_segs))
	
	FN2verbs, seen_verbs, unseen_verbs, train_seen_verb_instances, train_unseen_verb_instances, \
	test_seen_verb_instances, test_unseen_verb_instances = get_meta_info(eval_table)

	t = 'text_only' if opts.subtitle_only else 'text_visual'

	root = opts.result_dir
	output_dir = os.path.join(opts.result_dir, 'MAP', opts.subset, t)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	path = None
	
	for model, ckpt, model_ckpt in zip(models, ckpts, model_ckpts):
		if model == 'roberta':
			# hard coded
			path = f'{model}/MAP/{opts.subset}/results_{opts.split}/results_{t}_all.json'
		else:
			# hard coded
			# no [AFT] frame during evaluation, no_aft
			if opts.without_aft:
				path = f'{model}/MAP/{opts.subset}/results_{opts.split}/results_{ckpt}_{t}_no_aft_all.json'
			else:
				path = f'{model}/MAP/{opts.subset}/results_{opts.split}/results_{ckpt}_{t}_all.json'

		print(path)
		model_result_path = os.path.join(root, path)
		print('model checkpoint', model_ckpt)
		results = load_json(model_result_path)

		micro_acc = calculate_micro_acc(results)
		avg_seen_verb_acc, all_seen_verb_acc = calculate_macro_acc(seen_verbs, results)
		# test_seen_verb_instances = get_verb_num_instances(seen_verbs, results)
		avg_unseen_verb_acc, all_unseen_verb_acc = calculate_macro_acc(unseen_verbs, results)
		# test_unseen_verb_instances = get_verb_num_instances(unseen_verbs, results)
		harmonic_mean = get_harmonic_mean(avg_seen_verb_acc, avg_unseen_verb_acc)
		accuracies = {'micro_acc': micro_acc, 'seen_verb_acc': avg_seen_verb_acc,
		              'unseen_verb_acc': avg_unseen_verb_acc, 'harmonic_acc': harmonic_mean}
		
		model_output_dir = os.path.join(output_dir, model_ckpt)
		if not os.path.exists(model_output_dir):
			os.makedirs(model_output_dir)
		
		# overall seen verb acc, unseen verb acc
		for key in acc_type:
			df[model_ckpt][key] = accuracies.get(key)
		
		# something went wrong here, acc and num_instances
		if opts.output_per_verb_acc:
			for verb_type in ['seen_verb', 'unseen_verb']:
				print(verb_type)
				verb_acc = all_seen_verb_acc if verb_type == 'seen_verb' else all_unseen_verb_acc
				train_verb_instances = train_seen_verb_instances if verb_type == 'seen_verb' else train_unseen_verb_instances
				test_verb_instances = test_seen_verb_instances if verb_type == 'seen_verb' else test_unseen_verb_instances
				verb_stats = {verb: {'acc': verb_acc[verb],
				                     'train_num_instances': train_verb_instances[verb],
				                     'test_num_instances': test_verb_instances[verb]} for verb in verb_acc.keys()
				             }
				verb_acc_df = output_df(indexes=list(verb_stats.keys()), columns=['acc', 'train_num_instances', 'test_num_instances'], dict=verb_stats)
				print('Correlation of acc and training size: ', scipy.stats.spearmanr(verb_acc_df['train_num_instances'], verb_acc_df['acc']))
				print('Correlation of acc and test size: ', scipy.stats.spearmanr(verb_acc_df['test_num_instances'], verb_acc_df['acc']))

				verb_acc_df.iloc[:, [0]] = round(verb_acc_df.iloc[:, [0]].mul(100), 4)
				verb_acc_df.to_excel(os.path.join(model_output_dir, f'map_{model_ckpt}_{opts.split}_{verb_type}_acc.xlsx'))
				
		# copy the result file to the output directory
		dest_file_path = os.path.join(model_output_dir, f'results_{opts.split}.json')
		dest = shutil.copyfile(model_result_path, dest_file_path)
		
		# get wrong prediction video segment ids per model
		if opts.output_wrong_vid_seg_id:
			wrong_vid_seg_ids = get_wrong_vid_seg(results)
			filepath = os.path.join(model_output_dir, f'wrong_{opts.split}_{model_ckpt}.txt')
			save_lines(wrong_vid_seg_ids, filepath)
		
		# get seen verb/ unseen verb acc per FrameNet frame
		if opts.output_FN_acc:
			all_verb_acc = all_seen_verb_acc
			all_verb_acc.update(all_unseen_verb_acc)
			FN_frame_stats = get_FN_stats(FN2verbs, all_verb_acc)
			FN_frames = list(FN2verbs.keys())
		
			FN_per_model = output_df(indexes=FN_frames,
			                         columns=['seen_verbs_acc', 'unseen_verbs_acc',
			                                  'harmonic_acc', 'seen_verbs', 'unseen_verbs'],
			                         dict=FN_frame_stats)

			FN_per_model.iloc[:, [0, 1, 2]] = round(FN_per_model.iloc[:, [0, 1, 2]].mul(100), 4)
			FN_per_model.to_excel(os.path.join(model_output_dir, f'map_{model_ckpt}_{opts.split}_frame_acc.xlsx'))
			
	df.iloc[:, :] = df.iloc[:, :].mul(100)
	df.to_excel(os.path.join(output_dir, f'map_{opts.split}_acc.xlsx'))
	
	# output to latex tabel
	print("MAM Pretraining Intrinsic Evaluation Acc on MAP task")
	print(df.to_latex(index=True))
	print("\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# Required parameters
	parser.add_argument("--eval_table",
	                    default="/txt_db/exp_10_per/eval_table_10%.json",
	                    type=str,
	                    help="The eval table information that contain seen/unseen verb classes")
	parser.add_argument("--raw_data",
	                    default="all.json",
	                    type=str,
	                    help="raw data that contain video segment information")
	parser.add_argument("--video_segment_ids",
	                    default="/txt_db/exp_10_per/val_ids.json",
	                    type=str,
	                    help="video segment ids")
	parser.add_argument("--subset", default="10_per", type=str,
	                    help="The subset")
	parser.add_argument("--split", choices=["train", "val", "test"],
	                    default="val", type=str,
	                    help="The input evluation split")
	parser.add_argument("--models",
	                    default="verb_only,verb_random_joint,verb_random_alter", type=str,
	                    help="pretrained models")
	parser.add_argument("--checkpoints",
	                    default="10500,87500,98500", type=str,
	                    help="respective checkpoint steps of the pretrained models")
	parser.add_argument(
		"--result_dir", default=None, type=str,
		help="The input/output directory where the results will be retrieved/written.")
	parser.add_argument(
		"--output_per_verb_acc", action='store_true',
		help="To output accuracies by verb classes.")
	parser.add_argument(
		"--output_FN_acc", action='store_true',
		help="To output accuracies by FrameNet frames.")
	parser.add_argument(
		"--output_wrong_vid_seg_id", action='store_true',
		help="To output wrong video segment ids.")
	parser.add_argument('--subtitle_only',
	                    action='store_true',
	                    help="Whether to only have textual input present")
	parser.add_argument('--without_aft',
	                    action='store_true',
	                    help="Whether to leave out [AFT] frames in the visual input")
	args = parser.parse_args()
	main(args)
