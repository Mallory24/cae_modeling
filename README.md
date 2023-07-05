# cae_modeling
The official repository for creating casual action effect (CAE) modeling
-[TODO] Provide an overview

We built upon the HERO architecture with two pre-training tasks for implicit affordance learning:
Most codes are copied/modified from HERO 
The visual frame feature extraction (SlowFast + ResNet-152) also closely follows their recipe, which is available at [CAE_Video_Feature_Extractor](https://github.com/Mallory24/cae_video_feature_extractor)

## Create an environment
-[TO WRITE]

## Quick Start
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
1. download data
    ```bash
    # outside of the container
    bash scripts/download_tv_pretrain.sh $PATH_TO_STORAGE
    ```
2. pre-train
    ```bash
    # inside of the container
    horovodrun -np 16 python pretrain.py --config config/pretrain-tv-16gpu.json \
        --output_dir $PRETRAIN_EXP
    ```
    Unfortunately, we cannot host HowTo100M features due to its large size. Users can either process them on their own or send your inquiry to my email address (which you can find on our paper).
### MAM

### MEM

## Evaluate on intrinsic tasks
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
