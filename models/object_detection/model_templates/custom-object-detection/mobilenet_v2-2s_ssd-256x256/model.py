# model settings
input_size = 256
image_width, image_height = input_size, input_size
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True
    ),
    neck= None,
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels= (96, 320),
        stacked_convs=2,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200,
        nms_pre_classwise=200))
cudnn_benchmark = True
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.1),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/dobryaev/datasets/coco/annotations/instances_train2017.json',
            img_prefix='/home/dobryaev/datasets/coco/images/train2017',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file='/home/dobryaev/datasets/coco/annotations/instances_val2017.json',
        img_prefix='/home/dobryaev/datasets/coco/images/val2017',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/dobryaev/datasets/coco/annotations/instances_val2017.json',
        img_prefix='/home/dobryaev/datasets/coco/images/test2017',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[14, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/mobilenet_v2-2s_ssd-256x256_wo_neck'
load_from = None
resume_from = None
workflow = [('train', 1)]

'''
{96: ModuleList(
  (0): ConvModule(
    (conv): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (gn): GroupNorm(32, 32, eps=1e-05, affine=True)
    (activate): ReLU(inplace=True)
  )
  (1): ConvModule(
    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (gn): GroupNorm(32, 32, eps=1e-05, affine=True)
    (activate): ReLU(inplace=True)
  )
), 320: ModuleList(
  (0): ConvModule(
    (conv): Conv2d(320, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (gn): GroupNorm(32, 32, eps=1e-05, affine=True)
    (activate): ReLU(inplace=True)
  )
  (1): ConvModule(
    (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (gn): GroupNorm(32, 32, eps=1e-05, affine=True)
    (activate): ReLU(inplace=True)
  )
)}
'''