"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pretrain MEM Sampler
"""

import os.path
import json
import random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset, DataLoader
from toolz.sandbox import unzip
import math
import horovod.torch as hvd
from utils.logger import LOGGER

from .data import _check_ngpu


class MemNegativeSampler(BatchSampler):
    """Samples a batch_size of video segments grouped based on the same type of semantics.

    Args:
        vid2vidsegs/obj2vidsegs (dict): a dictionary of video id/object class to video segments ids.
        vidseg_ids (list): a list of video segments ids
        dataset (str): "train" or "val"
        batch_size (int): batch size.
    Return
        yields a batch of the video segment indices

    """
    
    def __init__(self, datasource, vidseg_ids, batch_size, dataset):
        self.datasource = datasource
        if _check_ngpu() > 1:
            self.keys = list(self.datasource.keys())[hvd.rank()::hvd.size()]
            # LOGGER.info(f'video ids: {len(self.keys)}')
        else:
            self.keys = list(self.datasource.keys())
            # LOGGER.info(f'video ids: {len(self.keys)}')

        # self.keys = list(self.datasource.keys())
        self.vidseg_ids = vidseg_ids
        self.batch_size = batch_size
        self.dataset = dataset
    
    def __iter__(self):
        batch = []
        if self.dataset == 'train':
            # a fixed batch size and a random order of indices
            for idx, k in enumerate(self.keys):
                remains = set(self.datasource.get(k))
                while len(remains) > self.batch_size:
                    selected = random.sample(list(remains), self.batch_size)
                    # get the idx of each video segment
                    selected_idx = [self.vidseg_ids.index(vidseg) for vidseg in selected]
                    batch.extend(selected_idx)
                    yield batch
                    batch = []
                    remains = remains - set(selected)
                
                sample_size = self.batch_size - len(remains)
                pool_vidsegs = list(set(self.vidseg_ids) - remains)
                selected = list(remains) + random.sample(list(pool_vidsegs), sample_size)
                # print(selected)
                selected_idx = [self.vidseg_ids.index(vidseg) for vidseg in selected]
                batch.extend(selected_idx)
                yield batch
                batch = []
        
        elif self.dataset in ['val', 'test']:
            # a dynamic batch size and a fixed order of indices
            for idx, k in enumerate(self.keys):
                remains = list(self.datasource.get(k))
                while len(remains) > self.batch_size:
                    # if there is more than one video segment per video
                    selected = remains[:self.batch_size]
                    # print(selected)
                    # get the idx of each video segment
                    selected_idx = [self.vidseg_ids.index(vidseg) for vidseg in selected]
                    batch.extend(selected_idx)
                    yield batch
                    batch = []
                    remains = remains[self.batch_size:]
                
                # for the video segment that do not have the corresponding negatives (i.e., remains < 1)
                # skip creation
                if len(remains) > 1:
                    selected = remains
                    selected_idx = [self.vidseg_ids.index(vidseg) for vidseg in selected]
                    batch.extend(selected_idx)
                    yield batch
                    batch = []


# For quick testing, use below:
class MemDataset(Dataset):
    def __init__(self, video_ids):
        self.ids = video_ids
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        vid = self.ids[i]
        return vid


class MetaLoader(object):
    """ wraps multiple data loader """
    
    def __init__(self, loaders, accum_steps=1, distributed=False):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)
        
        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0
    
    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
                if self.distributed:
                    # make sure all process is training same task
                    task = any_broadcast(task, 0)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_
            
            yield task, batch


def mem_collate(input):
    batch = {}
    vids = input
    batch['vids'] = vids
    return batch


def get_tiny_set(negative_dict, bsize):
    tiny_negative_dict = {}
    for k, v in negative_dict.items():
        if len(v) > bsize:
            if k not in tiny_negative_dict:
                new_v = v[:bsize * 2 + 3]
                tiny_negative_dict[k] = new_v
                break
    
    for k, v in negative_dict.items():
        if bsize < len(v) < 1:
            if k not in tiny_negative_dict:
                tiny_negative_dict[k] = v
                break
    
    for k, v in negative_dict.items():
        if len(v) == 1:
            if k not in tiny_negative_dict:
                tiny_negative_dict[k] = v
                break
    return tiny_negative_dict


def get_total_batches(negative_dict, dataset="train", bsize=5):
    batch_count = 0
    for k, v in negative_dict.items():
        if dataset == "train":
            if len(v) == 1:
                batch_count += 1
            else:
                batch_count += math.ceil(len(v) / bsize)
        else:
            if len(v) == 1:
                continue
            else:
                batch_count += math.ceil(len(v) / bsize)
    return batch_count


if __name__ == "__main__":
    random.seed(77)
    txt_db_folder = "/mount/arbeitsdaten/jp-silberer/ernie_vil/data/cae/single_result_verb_exp/42/txt_db"
    video_negatives = "train_video_based_negatives.json"
    object_negatives = "train_object_based_negatives.json"
    train_ids = "train_ids.json"
    
    vid2vidsegs_path = os.path.join(txt_db_folder, 'exp_10_per', 'neg', video_negatives)
    vid2vidsegs = json.load(open(vid2vidsegs_path, 'r'))
    
    vidseg_ids_path = os.path.join(txt_db_folder, 'exp_10_per', train_ids)
    vidseg_ids = json.load(open(vidseg_ids_path, 'r'))
    
    obj2vidsegs_path = os.path.join(txt_db_folder, 'exp_10_per', 'neg', object_negatives)
    obj2vidsegs = json.load(open(obj2vidsegs_path, 'r'))

    total_iter = get_total_batches(vid2vidsegs, dataset="train", bsize=64)
    print("video based train total_iter", total_iter)
    total_iter = get_total_batches(obj2vidsegs, dataset="val", bsize=64)
    print("object based train total_iter", total_iter)
    
    # Tiny: dummy dataset
    tiny_vid2vidsegs = get_tiny_set(vid2vidsegs, bsize=5)
    print(tiny_vid2vidsegs)
    
    total_iter = get_total_batches(tiny_vid2vidsegs, dataset="val", bsize=5)
    print("Tiny video based VAL total_iter", total_iter)
    
    val_dset = MemDataset(vidseg_ids)
    
    video_based_sampler = MemNegativeSampler(datasource=tiny_vid2vidsegs,
                                             vidseg_ids=vidseg_ids,
                                             batch_size=5,
                                             dataset="val")
    val_collate = mem_collate
    
    val_loader = DataLoader(val_dset,
                              batch_sampler=video_based_sampler,
                              num_workers=1, pin_memory=True,
                              collate_fn=val_collate)
    
    global_step = 0
    val_step = 10
    print("Normal Loader: ")
    for step, batch in enumerate(val_loader):
        global_step += 1
        print('step', step)
        print('batch vid seg ids', [vid_seg_id for vid_seg_id in batch['vids']])
        if global_step > val_step:
            break
    
    print("META Loader: ")
    # TODO: check if shuffle is implemented => over one go, the random selected vid is different
    train_dataloaders = {}
    train_loaders = {}
    train_loaders["mem"] = (train_loader, 1)
    train_dataloaders.update(train_loaders)
    meta_loader = MetaLoader(train_dataloaders, accum_steps=1, distributed=False)

    for step, (task, batch) in enumerate(meta_loader):
        print(f"Task: {task}")
        global_step += 1
        print('iter', step)
        print('batch vid seg ids', [vid_seg_id for vid_seg_id in batch['vids']])
        if global_step > train_step:
            break

