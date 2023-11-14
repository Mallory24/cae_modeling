"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from .data import (
    TxtTokLmdb, VideoFeatLmdb, SubTokLmdb,
    QueryTokLmdb, VideoFeatSubTokDataset, VideoFeatSingleSubTokDataset,
    video_collate, QaQueryTokLmdb)
from .loader import PrefetchLoader, MetaLoader
from .vcmr import (
    VcmrDataset, vcmr_collate, VcmrEvalDataset, vcmr_eval_collate,
    VcmrFullEvalDataset, vcmr_full_eval_collate)
from .vcmr_video_only import (
    VcmrVideoOnlyDataset, VcmrVideoOnlyEvalDataset,
    VcmrVideoOnlyFullEvalDataset)
from .vr_video_only import (
    VideoFeatDataset,
    VrVideoOnlyDataset, VrVideoOnlyEvalDataset,
    VrVideoOnlyFullEvalDataset)
from .vr import (
    VrDataset, VrEvalDataset, VrSubTokLmdb, VrQueryTokLmdb,
    MsrvttQueryTokLmdb,
    VrFullEvalDataset, vr_collate, vr_eval_collate,
    vr_full_eval_collate)
from .videoQA import (
    VideoQaDataset, video_qa_collate,
    VideoQaEvalDataset, video_qa_eval_collate)
from .violin import (
    ViolinDataset, violin_collate,
    ViolinEvalDataset, violin_eval_collate)
from .fom import (
    FomDataset, fom_collate,
    FomEvalDataset, fom_eval_collate)
from .vsm import VsmDataset, vsm_collate
from .mlm import (
    VideoMlmDataset, mlm_collate)
from .mfm import (
    MfmDataset, mfm_collate)
from .mam import (
    VideoMamDataset, mam_collate)
from .mem import (
    MemDataset, mem_collate)
# from .mem_sampler import (
#     MemNegativeSampler)
from .mem_sampler import (
    MemNegativeSampler)

# intrinsic evaluation of MAP: masked action prediction
from .map import (
    MAPFullEvalDataset, map_full_eval_collate, map_full_eval_collate_roberta)
# intrinsic evaluation of MEP: masked effect prediction
from .mep import (
    MEPFullEvalDataset, mep_full_eval_collate)

# downstream task evaluation
from .prost import (
    ProstDataset, prost_collate, prost_collate_roberta
)

from .tvc import TvcTrainDataset, TvcValDataset, CaptionTokLmdb
