_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/optical_sar/optical_sar_cloud_removal.py",
    "../../_base_/schedules/schedule_80k.py",
]
scale = 1
# model settings
module = "DynamicLSKBlock"
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="MoCoLSKNet",
        in_channels=1,
        gui_channels=2, # 双极化 sar vv vh
        num_feats=32,
        kernel_size=3,
        mocolsk_kernel_size=3,
        scale=scale,
        module=module,
        n_resblocks=4,
        num_stages=4,  # num of stages
        reduction=16,
        mlp_type="a",
        norm_flag=2,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=[0,0,0], std=[1,1,1], min=[0,0,0], max=[255,255,255]), # optical clear image norm paras.
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
    train_cfg=dict(),
    test_cfg=dict(
        metrics=[
            dict(type='SSIM'),
            dict(type="CC",),
            dict(type='MSE'),
            dict(type='PSNR'),
            dict(type='MAE'),
        ]
    ),
    data_preprocessor=dict(
        type="DataPreprocessor",
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ),
)
