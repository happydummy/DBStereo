# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

from .models.dbstereo.trainer import Trainer as DBStereoTrainer

'''
try:
# 'If you want to train/eval NMRF-Stereo, please refer to docs/prepare_foundationstereo.md
    from .models.foundationstereo.trainer import Trainer as FoundationStereoTrainer
except:
    raise ValueError('If you want to train/eval NMRF-Stereo, please refer to docs/prepare_foundationstereo.md. Otherwise you can comment out this line of code')
'''

# If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'
# try:
#     from .models.nmrf.trainer import Trainer as NMRFTrainer
# except:
#     raise ValueError("If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'")
__all__ = {
    'DBStereo': DBStereoTrainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
