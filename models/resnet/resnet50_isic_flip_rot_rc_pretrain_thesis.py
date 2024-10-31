# model settings

model_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),))

# dataloader pipeline
img_lighting_cfg = dict(
    eigval=[55.4625, 4.7940, 1.1475],
    eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]],
    alphastd=0.1,
    to_rgb=True)
policies = [
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=256,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Rotate',
        angle=30.0,
        #pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    #dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    #dict(type='Lighting', **img_lighting_cfg),
    dict(
        type='Normalize',
        mean=[195.461, 139.809, 146.056],
        std=[22.945, 30.287, 34.036],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        crop_size=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        mean=[195.461, 139.809, 146.056],
        std=[22.945, 30.287, 34.036],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size = 32,
    num_workers = 1,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = '/home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 200,
    ),
    test=dict(
        ckpt = '',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 32
# lr = 0.1 *32 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 32/256,
    #lr=0.001,
    momentum=0.9,
    weight_decay=1e-4)

# learning 
lr_config = dict(type='StepLrUpdater', step=[60, 120, 180])

# CUDA_VISIBLE_DEVICES=2 python tools/train_dm_online.py models/resnet/resnet50_isic_flip_rot_rc_pretrain_thesis.py 

# CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/resnet50_isic_flip_rot_rc_pretrain.py 

# CUDA_VISIBLE_DEVICES=1 python tools/train_resample.py models/resnet/resnet50_isic_flip_rot_rc_pretrain.py 

# CUDA_VISIBLE_DEVICES=3 python tools/train_resample_sqrt3.py models/resnet/resnet50_isic_flip_rot_rc_pretrain.py

# CUDA_VISIBLE_DEVICES=1 python tools/train_resample_pg.py models/resnet/resnet50_isic_flip_rot_rc_pretrain.py 

# CUDA_VISIBLE_DEVICES=1 python tools/train_dm.py /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet50_isic_flip_rot_rc_pretrain.py --aug_dir /home/slidm/DDPM/guided-diffusion/results/isic2018_64cs_050000iter_ddim50/samples_2000x64x64x3_s1.0.npz

# CUDA_VISIBLE_DEVICES=3 python tools/train_dm_online.py ./models/resnet/resnet50_isic_flip_rot_rc_pretrain.py

# CUDA_VISIBLE_DEVICES=5 python tools/train_dm_online_norm.py /home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet50_isic_flip_rot_rc.py

# CUDA_VISIBLE_DEVICES=7 python tools/train_dm_online_ug.py ./models/resnet/resnet50_isic_flip_rot_rc_pretrain.py