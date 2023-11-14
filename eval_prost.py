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
import os
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, FlavaForPreTraining

import torch.nn.functional as F

from data import (ProstDataset, prost_collate, prost_collate_roberta, PrefetchLoader)
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

from model.model import VideoModelConfig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_same_weights(prev_ckpt, current_ckpt):
    model1 = HeroForPretraining(
        VideoModelConfig(opts.model_config),
        vfeat_dim=VFEAT_DIM,  # 4352
        max_frm_seq_len=max_frm_seq_len,  # 100
        lw_neg_ctx=model_opts.lw_neg_ctx,
        lw_neg_q=model_opts.lw_neg_q,
        ranking_loss_type=model_opts.ranking_loss_type,
        use_hard_negative=False,
        hard_pool_size=model_opts.hard_pool_size,
        margin=model_opts.margin,
        use_all_neg=model_opts.use_all_neg)
    
    prev_model.load_partial_pretrained(prev_ckpt, VFEAT_DIM, max_frm_seq_len, skip_layers=True)

    model2 = HeroForPretraining.from_pretrained(
        model_config,
        state_dict=current_ckpt,
        vfeat_dim=VFEAT_DIM,
        max_frm_seq_len=max_frm_seq_len,
        lw_neg_ctx=model_opts.lw_neg_ctx,
        lw_neg_q=model_opts.lw_neg_q,
        ranking_loss_type=model_opts.ranking_loss_type,
        use_hard_negative=False,
        hard_pool_size=model_opts.hard_pool_size,
        margin=model_opts.margin,
        use_all_neg=model_opts.use_all_neg)
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def main(opts):
    # exit()
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
    
    device = torch.device("cuda")
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # Prepare model
    if opts.model_config == 'roberta':
        model = RobertaForMaskedLM.from_pretrained('roberta-base')

        inf_dataset = ProstDataset(opts.dataroot, for_roberta=True)
        eval_dataset = inf_dataset
        
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=prost_collate_roberta,
                                     shuffle=False)

        eval_dataloader = PrefetchLoader(eval_dataloader)
        
    elif opts.model_config == 'flava':
        model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
        inf_dataset = ProstDataset(opts.dataroot, for_flava=True)
        eval_dataset = inf_dataset

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=prost_collate_roberta,
                                     shuffle=False)

        eval_dataloader = PrefetchLoader(eval_dataloader)

    else:
        hps_file = f'{opts.output_dir}/log/hps.json'
        model_opts = Struct(load_json(hps_file))
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

        inf_dataset = ProstDataset(opts.dataroot)
        eval_dataset = inf_dataset
        
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=prost_collate,
                                     shuffle=False)

        eval_dataloader = PrefetchLoader(eval_dataloader)
        
    # check_same_weights(previous_ckpt, current_ckpt)

    model.to(device)

    if opts.fp16:
        model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
    
    is_roberta = True if opts.model_config == 'roberta' else False
    is_flava = True if opts.model_config == 'flava' else False

    results = validate_prost(model, eval_dataloader, is_roberta, is_flava)
    result_dir = f'{opts.output_dir}/PROST/'
    
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)
    
    if hvd.rank() == 0:
        save_json(
            results,
            f'{result_dir}/results_{opts.checkpoint}_all.json')
        LOGGER.info('All results written......')


@torch.no_grad()
def validate_prost(model, eval_loader, is_roberta, is_flava):
    LOGGER.info("start running PROST evaluation")
    model.eval()
    
    results = []
    for i, batch in enumerate(tqdm(eval_loader)):
        if is_roberta:
            mask_token_indexes = batch['mask_token_indexes']
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attn_masks'],
                            position_ids=batch['position_ids'])
            scores = outputs.logits[0, mask_token_indexes]
        if is_flava:
            # TO CHECK: for flava, what should be the input for "input_ids_masked"
            mask_token_indexes = batch['mask_token_indexes']
            outputs = model(input_ids=batch['input_ids'],
                            input_ids_masked=batch['input_ids'],
                            attention_mask=batch['attn_masks'],
                            position_ids=batch['position_ids'])
            scores = outputs.mlm_logits[0, mask_token_indexes]
        # reuse MAM head to perform inference
        else:
            scores = model(batch, task='mam', compute_loss=False)
            
        option_input_ids = batch['option_input_ids']

        logits_subset_options = torch.gather(scores, -1, option_input_ids)
        
        probs_masked_token = F.softmax(logits_subset_options, -1)

        pred_labels = torch.argmax(probs_masked_token, dim=1)
        # print("pred_labels", pred_labels)
        labels = batch['labels']
        # print("labels", labels)

        text_inputs = batch['text_inputs']
        label2tokens = batch['label2tokens']
        # print("label2tokens", label2tokens)
        groups = batch['groups']
        names = batch['names']

        logits_subset_options = [i for i in logits_subset_options.cpu().tolist()]
        probs = [i for i in probs_masked_token.cpu().tolist()]
        pred_labels = [i for i in pred_labels.cpu().tolist()]
        for pred_label, logit_subset, prob, label, text_input, label2token, group, name in \
                zip(pred_labels, logits_subset_options, probs, labels, text_inputs, label2tokens, groups, names):
            output_info = {}
            output_info['cloze question'] = text_input
            output_info['group'] = group
            output_info['name'] = name
            output_info['correct'] = pred_label == label
            output_info['label'] = label
            output_info['prediction'] = pred_label
            output_info['logits'] = logit_subset
            output_info['probabilities'] = prob
            output_info['label2token'] = label2token
            results.append(output_info)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="pretrained model config file")
    parser.add_argument("--batch_size",
                        default=64, type=int,
                        help="number of video segments in a batch")
    parser.add_argument("--dataroot",
                        default=None, type=str,
                        help="dataroot path to store cached data")
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
