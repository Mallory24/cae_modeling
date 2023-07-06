# cae_modeling
The official repository for creating casual action effect (CAE) modeling \
-[TODO] Provide an overview

We built upon the HERO architecture with two pre-training tasks for implicit affordance learning from the video domain: \
Most codes are copied/modified from [HERO's repository](https://github.com/linjieli222/HERO) \
The visual frame feature extraction (SlowFast + ResNet-152) also closely follows their recipe, which is available at [CAE_Video_Feature_Extractor](https://github.com/Mallory24/cae_video_feature_extractor)

## Create an environment
-[TO WRITE]

## Quick Start
-[TODO] 1. Write something about where they can download the pre-processed features.
To reproduce the whole dataset pre-processing steps, refer to **step 5**. \
Under **$PATH_TO_CAE/single_result_verb_exp/42/**, you should see the following folder structure:
```bash
    ├── cae.json
    ├── eval_table
    │   └── eval_table.json
    ├── test
    │   └── test.json
    ├── train
    │   └── train.json
    ├── val
    │   └── val.json
    ├── txt_db
    │    ├── full
    │    │   ├── data.mdb
    │    │   ├── eval_table.json
    │    │   ├── lock.mdb
    │    │   ├── meta.json
    │    │   ├── test_ids.json
    │    │   ├── train_ids.json
    │    │   ├── val_ids.json
    │    │   ├── vid2frame_sub_len.json
    │    │   ├── neg
    │    │   │    ├── train_video_based_negatives.json
    │    │   │    ├── val_10_video_based_negatives.json
    │    │   │    └── test_video_based_negatives.json
    └── video_db
        ├── full
        │    ├── id2nframe.json
        │    ├── resnet_slowfast_2.0_compressed
        │    │    ├── data.mdb
        │    │    ├── lock.mdb
        │    └── video_feat_info.pkl
```
-[TODO] 2. Run CAE multi-task pre-training 
-[TODO] Make the pre-trained checkpoints of (1) Multi-task training available
```bash
export PRETRAIN_EXP=model/cae/pretrain_cae_muliti
CUDA_VISIBLE_DEVICES=1,2 HOROVOD_CACHE_CAPACITY=0  horovodrun -np 2 python pretrain.py \
--config config/example-pretrain-cae-multi-2gpu.json \
--output_dir $PRETRAIN_EXP
```
Note: provide the checkpoints of RoBERTa-base under the field **_checkpoint_** in example_pretrain-cae-multi-2gpu.json

-[TODO] 3. Run intrinsic task evaluations: (1) Masked Action Prediction (MAP) (2) Masked Effect Prediction (MEP)
```bash
export PRETRAIN_EXP=pretrain_cae_multi
export CKPT=
export SPLIT=test

CUDA_VISIBLE_DEVICES=0 nohup horovodrun -np 1 python eval_map.py \
--sub_txt_db $PATH_TO_EXP/txt_db/$EXP_NAME \
--vfeat_db $PATH_TO_EXP/video_db/$EXP_NAME \
--split $SPLIT \
--subset $EXP_NAME \
--task cae_video_sub \
--checkpoint  $CKPT \
--model_config config/hero_finetune.json \
--batch_size 64 \
--output_dir $PRETRAIN_EXP 
```

-[TODO] 4. Run zero-shot evaluation on PROST.
```bash
```

Note: Feel free to replace `--output_dir` and `--checkpoint` with your own model trained in step 3.

**(Optional)** 5. To reproduce the CAE dataset preprocessing process. \
A. Create the CAE dataset from scratch: refer to this [repository](https://github.com/Mallory24/cae_dataset/blob/main/README.md). \
B. Extract visual features: refer to this [repository](https://github.com/Mallory24/cae_video_feature_extractor/blob/main/README.md). \
C. Collect visual features & save into lmdb: 
```bash
export $PATH_TO_EXP=path_to_single_result_verb_exp ($PATH_TO_CAE/single_result_verb_exp/42)
export $PATH_TO_STORAGE=path_to_videos
export $EXP_NAME=full 
```
Note: $EXP_NAME is arbitrary. 

Collect visual features:
```bash
python scripts/collect_video_feature_paths.py  \
--feature_dir $PATH_TO_STORAGE/videos/visual_features \
--output $PATH_TO_EXP/video_db/ \
--eval_table $PATH_TO_EXP/eval_table/eval_table.json  \
--dataset $EXP_NAME \
--nproc 0
```

Convert to lmdb:
```bash
python scripts/convert_videodb.py \
--vfeat_info_file $PATH_TO_EXP/video_db/$EXP_NAME/video_feat_info.pkl \
--output $PATH_TO_EXP/video_db \
--dataset $EXP_NAME \
--frame_length 2  \
--compress \
--clip_interval -1
```

After running the above codes, $PATH_TO_EXP/video_db/$EXP_NAME should have the following folder structure:
```bash
    ├── id2nframe.json
    ├── resnet_slowfast_2.0_compressed
    │    ├── data.mdb
    │    ├── lock.mdb
    └── video_feat_info.pkl
```

D. Tokenize subtitles & save into lmdb:
```bash
python scripts/prepro_sub.py \
--annotation $PATH_TO_EXP/cae.json \
--vid2nframe $PATH_TO_EXP/video_db/$EXP_NAME/id2nframe.json \
--eval_table $PATH_TO_EXP/eval_table/eval_table.json \
--frame_length 2 \
--output $PATH_TO_EXP/txt_db/$EXP_NAME \
--task cae
```

After running the above codes, $PATH_TO_EXP/txt_db/$EXP_NAME should have the following folder structure:
```bash
   ├── data.mdb
   ├── eval_table.json
   ├── lock.mdb
   ├── meta.json
   ├── test_ids.json
   ├── train_ids.json
   ├── val_ids.json
   └── vid2frame_sub_len.json
```

E. For the pretraining task MEM, get the video frame negatives for train/validation/test split:
```bash
export SPLIT=train
export IDS=train_ids.json 
export OPT=video-based # object-based
 
python scripts/get_negatives.py \
--annotation $PATH_TO_EXP/cae.json \
--vid2nframe $PATH_TO_EXP/video_db/$EXP_NAME/id2nframe.json \
--split $SPLIT \
--vid_idx $PATH_TO_EXP/txt_db/$EXP_NAME/$IDS \
--option $OPT \
--get_statistics \
--output_dir  $PATH_TO_EXP/txt_db/$EXP_NAME/neg
```

After running the above codes, $PATH_TO_EXP/txt_db/$EXP_NAME/neg/ should have the following folder structure:
```bash
   ├── train_video_based_negatives.json
   ├── val_video_based_negatives.json
   └── test_video_based_negatives.json
```
*NOTE*: We use val_10_video_based_negatives.json for quick validation to reduce the pretraining time.

## Pre-training
### Overview

### MAM
```bash
export PRETRAIN_EXP=model/cae/pretrain_cae_mam
CUDA_VISIBLE_DEVICES=1,2 HOROVOD_CACHE_CAPACITY=0  horovodrun -np 2 python pretrain.py \
--config config/example-pretrain-cae-mam-random-2gpu.json \
--output_dir $PRETRAIN_EXP
```
### MEM
```bash
export PRETRAIN_EXP=model/cae/pretrain_cae_mem
CUDA_VISIBLE_DEVICES=1,2 HOROVOD_CACHE_CAPACITY=0  horovodrun -np 2 python pretrain.py \
--config config/example-pretrain-cae-mem-random-2gpu.json \
--output_dir $PRETRAIN_EXP
```

## Zero-shot evaluation on PROST



## Citation
If you find this code useful for your research, please consider citing:
```bibtex
@inproceedings{li2020hero,
  title={HERO: Hierarchical Encoder for Video+ Language Omni-representation Pre-training},
  author={Li, Linjie and Chen, Yen-Chun and Cheng, Yu and Gan, Zhe and Yu, Licheng and Liu, Jingjing},
  booktitle={EMNLP},
  year={2020}
}
```

## License

-[TODO]
