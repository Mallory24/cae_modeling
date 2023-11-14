import ujson as json
import random
import os
import pandas as pd
import pprint
import argparse

def get_vid2vidseg(data):
	vid2vidseg = {}
	for vidseg in data:
		vid = vidseg.split("_")[:-1]
		if len(vid) > 1:
			vid = "_".join(vid)
		else:
			vid = vid[0]
		if vid not in vid2vidseg:
			vid2vidseg[vid] = set()
			vid2vidseg[vid].add(vidseg)
		vid2vidseg[vid].add(vidseg)
	for vid in vid2vidseg:
		vid2vidseg[vid] = list(vid2vidseg[vid])
	return vid2vidseg

def get_vid2nframes(data, id2nframe):
	vid2nframe = {}
	for vidseg in data:
		vid = vidseg.split("_")[0]
		if vid not in vid2nframe:
			vid2nframe[vid] = 0
		vid2nframe[vid] += id2nframe.get(vidseg)
	return vid2nframe


def get_vid2AFTframes(data, id2nframe):
	vid2AFTframe = {}
	for vidseg in data:
		vid = vidseg.split("_")[0]
		if vid not in vid2AFTframe:
			vid2AFTframe[vid] = 0
		AFTframes = (id2nframe.get(vidseg) // 3) * 1
		vid2AFTframe[vid] += AFTframes
	return vid2AFTframe


def get_object2vidseg(data, annotation):
	object2vidseg = {}
	for vidseg in data:
		annot = annotation.get(vidseg)
		object_set = annot.get('nouns')
		for obj in object_set:
			if obj not in object2vidseg:
				object2vidseg[obj] = set()
			object2vidseg[obj].add(vidseg)
	for obj in object2vidseg:
		object2vidseg[obj] = list(object2vidseg[obj])
	return object2vidseg


def get_object2nframes(object2vidseg, id2nframe):
	obj2nframe = {}
	for obj, vidseg_set in object2vidseg.items():
		for vidseg in vidseg_set:
			nframes = id2nframe.get(vidseg)
			if obj not in obj2nframe:
				obj2nframe[obj] = 0
			obj2nframe[obj] += nframes
	return obj2nframe


def get_object2AFTframes(object2vidseg, id2nframe):
	obj2AFTframe = {}
	for obj, vidseg_set in object2vidseg.items():
		for vidseg in vidseg_set:
			nframes = id2nframe.get(vidseg)
			AFTframes = (nframes // 3) * 1
			if obj not in obj2AFTframe:
				obj2AFTframe[obj] = 0
			obj2AFTframe[obj] += AFTframes
	return obj2AFTframe


def get_vidseg2neg_counts(data, vidseg2nframe, vid2AFTframe, obj2AFTframe, annotation, option='video'):
	vidseg2negatives = {}
	for vidseg in data:
		nframes = vidseg2nframe.get(vidseg)
		if option == 'video':
			vid = vidseg.split("_")[0]
			# total nframes (negatives) - the current nframes [AFT] frame part
			vidseg2negatives[vidseg] = vid2AFTframe.get(vid) - ((nframes // 3) * 1)
		elif option == 'object':
			annot = annotation.get(vidseg)
			object_set = annot.get('nouns')
			total_negatives = 0
			if len(object_set) == 0:
				vidseg2negatives[vidseg] = 0
			else:
				for obj in object_set:
					total_negatives += obj2AFTframe.get(obj)
				neg_frames = total_negatives - ((nframes // 3) * 1)
				vidseg2negatives[vidseg] = neg_frames
	return vidseg2negatives


def calculate_video_based_neg(vidseg_ids, vidseg2nframe, dataset):
	stats = {}

	# get a dictionary of a video to a list of video segments
	vid2vidseg = get_vid2vidseg(vidseg_ids)
	vidsegs_counts = [len(vidsegs) for vid, vidsegs in vid2vidseg.items()]
	avg_vidseg = sum(vidsegs_counts) / len(vidsegs_counts)
	max_vidseg = max(vidsegs_counts)
	min_vidseg = min(vidsegs_counts)
	stats['vid2vidseg'] = {'avg': round(avg_vidseg, 2), 'max': max_vidseg, 'min': min_vidseg}
	
	# get the total [AFT] frames per video
	vid2AFTframe = get_vid2AFTframes(vidseg_ids, vid2nframe)
	nframe_counts = [nframe for nframe in vid2AFTframe.values()]
	avg_nframes = sum(nframe_counts) / len(nframe_counts)
	max_nframes = max(nframe_counts)
	min_nframes = min(nframe_counts)
	stats['vid2AFTframe'] = {'avg': round(avg_nframes, 2), 'max': max_nframes, 'min': min_nframes}
	
	# iterate through video segments
	vidseg2negatives = get_vidseg2neg_counts(vidseg_ids,
	                                         vidseg2nframe=vidseg2nframe,
	                                         vid2AFTframe=vid2AFTframe,
	                                         obj2AFTframe=None,
	                                         annotation=None,
	                                         option='video')
	
	negative_counts = [neg for neg in vidseg2negatives.values()]
	zero_negative_counts = [neg for neg in vidseg2negatives.values() if neg == 0]
	
	avg_negs = sum(negative_counts) / len(negative_counts)
	max_negs = max(negative_counts)
	min_negs = min(negative_counts)
	
	stats['vidseg2neg'] = {'avg': round(avg_negs, 2), 'max': max_negs, 'min': min_negs}
	stats['total video segments without negatives'] = len(zero_negative_counts)
	
	pprint.pprint(dataset)
	pprint.pprint(stats)


def calculate_object_based_neg(vidseg_ids, vidseg2nframe, dataset):
	stats = {}

	# get the dictionary of object to a set of video segment
	obj2vidseg = get_object2vidseg(vidseg_ids, annotation)
	# print("total unique objects", len(obj2vidseg.keys()))
	vidsegs_counts = [len(vidsegs) for vidsegs in obj2vidseg.values()]
	avg_vidseg = sum(vidsegs_counts) / len(vidsegs_counts)
	max_vidseg = max(vidsegs_counts)
	min_vidseg = min(vidsegs_counts)
	stats['unique_objects'] = len(obj2vidseg.keys())
	stats['obj2vidseg'] = {'avg': round(avg_vidseg, 2), 'max': max_vidseg, 'min': min_vidseg}
		
	# get the total [AFT] frames per object
	obj2AFTframe = get_object2AFTframes(obj2vidseg, vid2nframe)
	nframe_counts = [nframe for nframe in obj2AFTframe.values()]
	avg_nframes = sum(nframe_counts) / len(nframe_counts)
	max_nframes = max(nframe_counts)
	min_nframes = min(nframe_counts)
	stats['obj2AFTframe'] = {'avg': round(avg_nframes, 2), 'max': max_nframes, 'min': min_nframes}
	
	# iterate through video segments, and identify its object,
	# then gather the count of the negative
	vidseg2negatives = get_vidseg2neg_counts(vidseg_ids,
	                                         vidseg2nframe=vidseg2nframe,
	                                         vid2AFTframe=None,
	                                         obj2AFTframe=obj2AFTframe,
	                                         annotation=annotation,
	                                         option='object')
	
	negative_counts = [neg for neg in vidseg2negatives.values()]
	zero_negative_counts = [neg for neg in vidseg2negatives.values() if neg == 0]
	
	avg_negs = sum(negative_counts) / len(negative_counts)
	max_negs = max(negative_counts)
	min_negs = min(negative_counts)
	
	stats['vidseg2neg'] = {'avg': round(avg_negs, 2), 'max': max_negs, 'min': min_negs}
	stats['total video segments without negatives'] = len(zero_negative_counts)
	
	pprint.pprint(dataset)
	pprint.pprint(stats)


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation', required=True, default='cae.json', help='annotation JSON')
	parser.add_argument('--vid2nframe', type=str, default="id2nframe.json", help="vid info file")
	parser.add_argument('--split', type=str, default="train", help="the split of the dataset")
	parser.add_argument('--vid_idx', type=str, default="train_ids.json", help="video segment ids")
	parser.add_argument("--option", choices=["video-based", "object-based"], default="video", required=True,
	                    help="choose a strategy to collect negatives")
	parser.add_argument("--get_statistics", action="store_true", default=False,
	                    help="whether to output the statistics of average negatives per video segment")
	parser.add_argument("--output_dir", type=str, default="neg", required=True,
	                    help="the output directory of the negatives")
	
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	option = args.option
	vid_ids = json.load(open(args.vid_idx))
	vid2nframe = json.load(open(args.vid2nframe))
	annotation = json.load(open(args.annotation))
	output_dir = args.output_dir
	split = args.split
	get_statistics = args.get_statistics
	
	if option == 'video-based':
		if get_statistics:
			calculate_video_based_neg(vid_ids, vid2nframe, split)
		neg = get_vid2vidseg(vid_ids)
		output_file = f'{split}_video_based_negatives.json'
	
	elif option == 'object-based':
		if get_statistics:
			calculate_object_based_neg(vid_ids, vid2nframe, args.split)
		neg = get_object2vidseg(vid_ids, annotation)
		output_file = f'{split}_object_based_negatives.json'
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	output_filepath = os.path.join(output_dir, output_file)
	with open(output_filepath, "w") as f:
		f.write(json.dumps(neg, indent=4))
