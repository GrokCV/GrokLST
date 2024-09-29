# training schedule for 10k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=1000)
val_cfg = dict(type="MultiValLoop")
test_cfg = dict(type="MultiTestLoop")

# learning policy
param_scheduler = dict(
    type="CosineRestartLR", by_epoch=False, periods=[2500, 2500, 2500, 2500], restart_weights=[1, 1, 1, 1], eta_min=1e-7
)

# optimizer
optim_wrapper = dict(
    constructor="DefaultOptimWrapperConstructor",
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-5),
)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=4)

# base_lr = 1.0
# optim_wrapper = dict(
#     # _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='DAdaptAdam', lr=base_lr, weight_decay=0.05,
#         decouple=True),
#     paramwise_cfg=dict(
#         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
