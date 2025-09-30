# optimizer
lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

# learning-rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[60, 75, 80],
        gamma=0.2)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop') 