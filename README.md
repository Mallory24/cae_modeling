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
To reproduce the whole dataset pre-processing steps, refer to step 5. \
under **$PATH_TO_CAE/single_result_verb_exp/42/**, you should see the following folder structure:
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
    │    │   └── vid2frame_sub_len.json   
    └── video_db
        ├── full
        │    ├── id2nframe.json
        │    ├── resnet_slowfast_2.0_compressed
        │    │    ├── data.mdb
        │    │    ├── lock.mdb
        │    └── video_feat_info.pkl
```
-[TODO] 2. Run (multi-task) pre-training, add an in-document reference link for other pre-training strategies

-[TODO] Make the pretrained checkpoints of (1) Multi-task training available
```bash
```

-[TODO] 3. Run intrinsic task evaluations: (1) Masked Action Prediction (2) Masked Effect Prediction
```bash
```
-[TODO] 4. Run zero-shot evaluation on PROST.
```bash
```

Note: Feel free to replace `--output_dir` and `--checkpoint` with your own model trained in step 3.

(Optional) 5. To reproduce the CAE dataset preprocessing process. \
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

## (HERO) Quick Start
*NOTE*: Please run `bash scripts/download_pretrained.sh $PATH_TO_STORAGE` to get our latest pretrained
checkpoints.

We use TVR as an end-to-end example for using this code base.

1. Download processed data and pretrained models with the following command.
    ```bash
    bash scripts/download_tvr.sh $PATH_TO_STORAGE
    ```
    After downloading you should see the following folder structure:
    ```
    ├── finetune
    │   ├── tvr_default
    ├── video_db
    │   ├── tv
    ├── pretrained
    │   └── hero-tv-ht100.pt
    └── txt_db
        ├── tv_subtitles.db
        ├── tvr_train.db
        ├── tvr_val.db
        └── tvr_test_public.db
    ```

2. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/video_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/src` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)


3. Run finetuning for the TVR task.
    ```bash
    # inside the container
    horovodrun -np 8 python train_vcmr.py --config config/train-tvr-8gpu.json

    # for single gpu
    python train_vcmr.py --config $YOUR_CONFIG_JSON
    ```

4. Run inference for the TVR task.
    ```bash
    # inference, inside the container
    horovodrun -np 8 python eval_vcmr.py --query_txt_db /txt/tvr_val.db/ --split val \
        --vfeat_db /video/tv/ --sub_txt_db /txt/tv_subtitles.db/ \
        --output_dir /storage/tvr_default/ --checkpoint 4800 --fp16 --pin_mem

    ```
    The result file will be written at `/storage/tvr_default/results_val/results_4800_all.json`.
    Change to  ``--query_txt_db /txt/tvr_test_public.db/ --split test_public`` for inference on test_public split.
    Please format the result file as requested by the evaluation server for submission, our code does not include formatting.

    The above command runs inference on the model we trained.
    Feel free to replace `--output_dir` and `--checkpoint` with your own model trained in step 3.
    Single GPU inference is also supported.


5. Misc.
In case you would like to reproduce the whole preprocessing pipeline.

* Text annotation and subtitle preprocessing
    ```bash
    # outside of the container
    bash scripts/create_txtdb.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/ann
    ```

* Video feature extraction

    We provide feature extraction code at [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor).
    Please follow the link for instructions to extract both 2D ResNet features and 3D SlowFast features.
    These features are saved as separate .npz files per video.

* Video feature preprocessing and saved to lmdb
    ```bash
    # inside of the container

    # Gather slowfast/resnet feature paths
    python scripts/collect_video_feature_paths.py  --feature_dir $PATH_TO_STORAGE/feature_output_dir\
        --output $PATH_TO_STORAGE/video_db --dataset $DATASET_NAME
    
    # Convert to lmdb
    python scripts/convert_videodb.py --vfeat_info_file $PATH_TO_STORAGE/video_db/$DATASET_NAME/video_feat_info.pkl \
        --output $PATH_TO_STORAGE/video_db --dataset $DATASET_NAME --frame_length 1.5
    ```
    - `--frame_length`: 1 feature per "frame_length" seconds, we use 1.5/2 in our implementation. set it to be consistent with the one used in feature extraction.
    - `--compress`: enable compression of lmdb

## Pre-training
### Overview

### MAM
1. verb_random_joint

### MEM

## Evaluation on intrinsic tasks
### MAP

### MEP


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
