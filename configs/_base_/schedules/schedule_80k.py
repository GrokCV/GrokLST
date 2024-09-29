# training schedule for 10k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=4000)
val_cfg = dict(type="MultiValLoop")
test_cfg = dict(type="MultiTestLoop")

# learning policy, i.e., torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
# https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html
# https://mmengine.readthedocs.io/en/latest/_modules/mmengine/optim/scheduler/lr_scheduler.html

param_scheduler = dict(
    type="CosineRestartLR", by_epoch=False, periods=[80000], restart_weights=[1], eta_min=1e-7
)
# param_scheduler = dict(
#     type="CosineRestartLR", by_epoch=False, periods=[40000, 40000], restart_weights=[1, 0.1], eta_min=1e-7
# )
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

# base_lr = 0.0001
# optim_wrapper = dict(
#     # _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='DAdaptAdam', lr=base_lr, weight_decay=0.05,
#         decouple=True),
#     paramwise_cfg=dict(
#         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
