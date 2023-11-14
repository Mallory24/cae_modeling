"""
gather slowfast/resnet feature paths
"""
import os
import numpy as np
import pickle as pkl
import argparse
from tqdm import tqdm
from cytoolz import curry
import multiprocessing as mp
import json


def load_npz_full(slowfast_f, slowfast_dir, resnet_dir):
    vid = slowfast_f.split("/")[-1].split(".npz")[0]
    folder_name = slowfast_f.split("/")[-3]
    
    resnet_f = slowfast_f.replace(slowfast_dir, resnet_dir)
    try:
        slowfast_data = np.load(slowfast_f, allow_pickle=True)
        slowfast_frame_len = max(0, len(slowfast_data["features"]))
        # print(f"slowfast frame len {slowfast_frame_len}")
    except Exception:
        slowfast_frame_len = 0
        
    resnet_frame_len = 0
    if slowfast_frame_len == 0:
        slowfast_f = ""
        print(f"Corrupted slowfast files for {vid}")
        
    if not os.path.exists(resnet_f):
        resnet_f = ""
        print(f"resnet files for {vid} does not exists")
    else:
        try:
            resnet_data = np.load(resnet_f, allow_pickle=True)
            resnet_frame_len = len(resnet_data["features"])
            # print(f"resnet_frame_len {resnet_frame_len}")
        except Exception:
            resnet_frame_len = 0
            resnet_f = ""
            print(f"Corrupted resnet files for {vid}")
    frame_len = min(slowfast_frame_len, resnet_frame_len)   # frame_len =>  n_chunks
    return vid, frame_len, slowfast_f, resnet_f, folder_name


@curry
def load_npz(slowfast_f, slowfast_dir, resnet_dir):
    vid = slowfast_f.split("/")[-1].split(".npz")[0]

    folder_name = slowfast_f.split("/")[-2]

    resnet_f = slowfast_f.replace(slowfast_dir, resnet_dir)
    try:
        slowfast_data = np.load(slowfast_f, allow_pickle=True)
        slowfast_frame_len = max(0, len(slowfast_data["features"]))
        print(f"slowfast frame len {slowfast_frame_len}")
    except Exception:
        slowfast_frame_len = 0
    resnet_frame_len = 0
    if slowfast_frame_len == 0:
        slowfast_f = ""
        print(f"Corrupted slowfast files for {vid}")
    # print(resnet_f)
    if not os.path.exists(resnet_f):
        resnet_f = ""
        print(f"resnet files for {vid} does not exists")
    else:
        try:
            resnet_data = np.load(resnet_f, allow_pickle=True)
            resnet_frame_len = len(resnet_data["features"])
        except Exception:
            resnet_frame_len = 0
            resnet_f = ""
            print(f"Corrupted resnet files for {vid}")
    frame_len = min(slowfast_frame_len, resnet_frame_len)   # frame_len =>  n_chunks
    return vid, frame_len, slowfast_f, resnet_f, folder_name


def extract_video_segment_ids(eval_tabel):
    eval = json.load(open(eval_tabel, "r"))
    vid_seg_ids = set()
    for F in eval.keys():
        for verb_cls in eval[F].keys():
            for v in eval[F][verb_cls]:
                for split in eval[F][verb_cls][v].keys():
                    for d in eval[F][verb_cls][v][split].keys():
                        vid_segs = eval[F][verb_cls][v][split][d]
                        vid_seg_ids.update(set(vid_segs))
    return vid_seg_ids


def main(opts):
    
    vid_seg_ids = extract_video_segment_ids(opts.eval_table)
    slowfast_dir = os.path.join(opts.feature_dir, "slowfast_features/")
    resnet_dir = os.path.join(opts.feature_dir, "resnet_features/")

    loaded_file = []
    for root, dirs, curr_files in os.walk(f'{slowfast_dir}/'):
        for f in curr_files:
            if f.endswith('.npz'):
                if f[:-4] in vid_seg_ids:
                    slowfast_f = os.path.join(root, f)
                    loaded_file.append(slowfast_f)
    print(f"Found {len(loaded_file)} slowfast files....")
    print(f"sample loaded_file: {loaded_file[:3]}")
    failed_resnet_files, failed_slowfast_files = [], []
    files = {}
    
    if opts.nproc == 0:     # run sequentially
        for slowfast_f in tqdm(loaded_file):
            vid, frame_len, slowfast_f, resnet_f, folder_name = load_npz_full(slowfast_f, slowfast_dir, resnet_dir)
            files[vid] = (frame_len, slowfast_f, resnet_f, folder_name)

            if resnet_f == "":
                video_file = os.path.join(folder_name, vid)
                failed_resnet_files.append(video_file)
            if slowfast_f == "":
                video_file = os.path.join(folder_name, vid)
                failed_slowfast_files.append(video_file)
    else:
        load = load_npz(slowfast_dir, resnet_dir)
        with mp.Pool(opts.nproc) as pool, tqdm(total=len(loaded_file)) as pbar:
            for i, (vid, frame_len, slowfast_f,
                    resnet_f, folder_name) in enumerate(
                    pool.imap_unordered(load, loaded_file, chunksize=len(loaded_file) // opts.nproc)):  # chunksize = 128, imap_unordered
                # imap can only take one varying argument
                files[vid] = (frame_len, slowfast_f, resnet_f, folder_name)
                if resnet_f == "":
                    video_file = os.path.join(folder_name, vid)
                    failed_resnet_files.append(video_file)
                if slowfast_f == "":
                    video_file = os.path.join(folder_name, vid)
                    failed_slowfast_files.append(video_file)
                pbar.update(1)
            
    output_dir = os.path.join(opts.output, opts.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    pkl.dump(files, open(os.path.join(
        output_dir, "video_feat_info.pkl"), "wb"))
    if len(failed_slowfast_files):
        pkl.dump(failed_slowfast_files, open(os.path.join(
            output_dir, "failed_slowfast_files.pkl"), "wb"))
    if len(failed_resnet_files):
        pkl.dump(failed_resnet_files, open(os.path.join(
            output_dir, "failed_resnet_files.pkl"), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir",
                        default="",
                        type=str, help="The input video feature dir.")
    parser.add_argument("--output", default=None, type=str,
                        help="output dir")
    parser.add_argument('--dataset', type=str,
                        default="")
    parser.add_argument('--eval_table', type=str,
                        default="eval_table_10%.json")
    parser.add_argument('--nproc', type=int, default=4,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)

    