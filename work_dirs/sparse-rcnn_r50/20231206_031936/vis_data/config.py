auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 8e-05
data_root = 'SKU110K_fixed/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'checkpoints/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 10
metainfo = dict(
    classes=('object', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.5,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.5,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.5,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.5,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.5,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=5,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
        ],
        type='SparseRoIHead'),
    rpn_head=dict(
        num_proposals=200,
        proposal_feature_channel=256,
        type='EmbeddingRPNHead'),
    test_cfg=dict(rcnn=dict(max_per_img=200), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
        ],
        rpn=None),
    type='SparseRCNN')
num_proposals = 200
num_stages = 5
optim_wrapper = dict(
    optimizer=dict(lr=8e-05, type='AdamW', weight_decay=0.02),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=10, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=5,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        eta_min=4.000000000000001e-06,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(
    max_per_img=200,
    min_bbox_size=0,
    nms=dict(iou_threshold=0.45, type='nms'),
    nms_pre=1000,
    score_thr=0.05,
    type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco_annotations_val.json',
        backend_args=None,
        data_prefix=dict(img='images/val/'),
        data_root='SKU110K_fixed/',
        metainfo=dict(classes=('object', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='SKU110K_fixed/coco_annotations_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_batch_size_per_gpu = 4
train_cfg = (dict(
    allowed_border=-1,
    assigner=dict(
        ignore_iof_thr=-1,
        min_pos_iou=0,
        neg_iou_thr=0.35,
        pos_iou_thr=0.45,
        type='MaxIoUAssigner'),
    debug=False,
    pos_weight=-1,
    sampler=dict(type='PseudoSampler')), )
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='coco_annotations_train.json',
        backend_args=None,
        data_prefix=dict(img='images/train/'),
        data_root='SKU110K_fixed/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('object', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 2
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco_annotations_val.json',
        backend_args=None,
        data_prefix=dict(img='images/val/'),
        data_root='SKU110K_fixed/',
        metainfo=dict(classes=('object', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='SKU110K_fixed/coco_annotations_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs\\sparse-rcnn_r50'
