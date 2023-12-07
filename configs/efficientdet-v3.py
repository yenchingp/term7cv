_base_ = '../mmdetection/configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py'

data_root = 'SKU110K_fixed/'

train_batch_size_per_gpu = 4
train_num_workers = 2
# set max no of detected boxes
num_proposals = 200
num_stages = 5

no_of_classes = 1
max_epochs = 10
base_lr = 0.00008
neg_iou_thr = 0.4
norm_cfg = dict(type='BN', requires_grad=True)

metainfo = {
    'classes': ('object', ),
    'palette': [
        (220, 20, 60),
    ]
}

model = dict(
    bbox_head=dict(
        type='RetinaSepBNHead',
        num_ins=5,
        norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(neg_iou_thr=neg_iou_thr),
        optimizer=dict(type='Adam', lr=base_lr)  # Adjust the lr value here
    )
)

model = dict(
    bbox_head=dict(
        type='RetinaSepBNHead', 
        num_ins=5, 
        norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='coco_annotations_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/val/'),
        ann_file='coco_annotations_val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'coco_annotations_val.json',
)

test_evaluator = val_evaluator


# linear learning rate schedule for the first 10 iterations, 
# and then switches to a cosine annealing learning rate schedule from the midpoint
# of the training (epoch max_epochs // 2) until the end of the training (epoch max_epochs). 
# The cosine annealing schedule gradually reduces the learning rate to eta_min over the specified number of epochs (T_max).

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# reduce learning rate of optimizer, stick with SGD for image classification
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.02),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# customize checkpoints to save weights and log it
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

# change to custom directory for weights
load_from = 'checkpoints/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth'

# reduce max training epochs from 12 to 10
train_cfg = dict( 
    max_epochs=max_epochs)

# replace to use TensorBoard as Visualiser
visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='TensorboardVisBackend')])
