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
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Invert', prob=0.5),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    dict(
        type='Posterize',
        magnitude_key='bits',
        magnitude_range=(0, 4),
        prob=0.5),
    dict(
        type='Solarize',
        magnitude_key='thr',
        magnitude_range=(0, 256),
        prob=0.5),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110),
        thr=128,
        prob=0.5),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Contrast',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Brightness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.5,
        random_negative_prob=0.),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5),
    dict(
        type='Cutout',
        magnitude_key='shape',
        magnitude_range=(1, 41),
        pad_val=0,
        prob=0.5),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='horizontal',
        random_negative_prob=0.5,
        interpolation='bicubic'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.5,
        direction='vertical',
        random_negative_prob=0.5,
        interpolation='bicubic')
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
        size=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        mean=[195.461, 139.809, 146.056],
        std=[22.945, 30.287, 34.036],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

# train
data_cfg = dict(
    batch_size = 1,
    num_workers = 1,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = '',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 200,
    ),
    test=dict(
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones/logs/ResNet/2024-01-31-15-55-19/Val_Epoch125-Acc76.396.pth',
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones-1/logs/ResNet/2024-02-09-15-12-28/Val_Epoch139-Acc90.121.pth', # our (test best)
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones-1/logs/ResNet/2024-02-09-15-12-28/Train_Epoch110-Loss0.087.pth', # our (train best)
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones-1/logs/ResNet/2024-05-14-16-34-02/Train_Epoch108-Loss0.079.pth', # our rebuttal (train best)
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones-1/logs/ResNet/2024-05-14-16-34-02/Val_Epoch177-Acc90.423.pth', # our rebuttal (test best)
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones-1/logs/ResNet/2024-02-10-23-21-27/Val_Epoch129-Acc89.567.pth', # CE
        #ckpt = '/home/slidm/OCTA/Awesome-Backbones/models/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        ckpt = '/home/slidm/OCTA/Awesome-Backbones/logs/ResNet/2024-10-17-21-30-15/Last_Epoch200.pth',
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

# CUDA_VISIBLE_DEVICES=3 python tools/evaluation.py models/resnet/resnet50_isic_flip_rot_rc_eval.py 

# 