default_scope = "mmagic"
save_dir = "./work_dirs"

#default_hooks = dict(
#    timer=dict(type="IterTimerHook"),
#    logger=dict(type="LoggerHook", interval=100),
#    param_scheduler=dict(type="ParamSchedulerHook"),
#    checkpoint=dict(
#        type="CheckpointHook",
#        interval=1,
#        out_dir=save_dir,
#        by_epoch=False,
#        max_keep_ckpts=1,
#        save_best="auto",
 #       rule="greater",
 #   ),
#    sampler_seed=dict(type="DistSamplerSeedHook"),
#)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1000,
        save_optimizer=False,
        by_epoch=False,
        out_dir=None,
        save_best="auto",
        max_keep_ckpts=1,
        rule="less",
        save_last=False,
    ),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

model_wrapper_cfg = dict(
    type="MMSeparateDistributedDataParallel",
    broadcast_buffers=False,
    find_unused_parameters=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=4),
    dist_cfg=dict(backend="nccl"),
)

log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=100, by_epoch=False)

load_from = None
resume = False

#vis_backends = [dict(type="LocalVisBackend")]
#visualizer = dict(
#    type="ConcatImageVisualizer",
#    vis_backends=vis_backends,
 #   fn_key="hr_lst_path",
    # img_keys=["pred_img"],  # only save pred_img
#    img_keys=[
#        "input",
#        "pred_img",
#        "gt_img",
#    ],  # for concating
#    bgr2rgb=True,
#)

#custom_hooks = [dict(type="BasicVisualizationHook", interval=1, on_val=False, on_test=True)]
vis_backends = [dict(type="LocalVisBackend")]  # not use
visualizer = dict(
    type="ConcatLSTVisualizer",
    vis_backends=vis_backends,
    fn_key="hr_lst_path",
    # img_keys=["pred_img"],  # only save pred_img
    img_keys=[
        "input",
        "pred_img",
        "gt_img",
    ],  # for concating
    mask_keys=["lr_mask", "hr_mask", "hr_mask"],
)

custom_hooks = [dict(type="BasicVisualizationHook", interval=1, on_val=False, on_test=True)]
