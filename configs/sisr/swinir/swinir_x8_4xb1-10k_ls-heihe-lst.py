_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/large_scale_heihe_lst/ls_heihe_lst_x8-64-512_sisr.py",
    "../../_base_/schedules/schedule_20k.py",
]

scale = 8
img_size = 64
# model settings
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="SwinIR",
        upscale=scale,  # 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction.
        in_chans=1,
        img_size=img_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        norm_flag=1,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=278.33, std=15.27, min=242.23, max=324.78),  # 240m paras
    ),
    pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
    train_cfg=dict(),
    test_cfg=dict(
        metrics=[
            dict(type="RMSE", scaling=1.0, prefix="lst"),
            dict(type="LSTMAE", scaling=1.0, prefix="lst"),  # abs(pred-gt)
            dict(type="BIAS", scaling=1.0, prefix="lst"),  # pred-gt
            dict(type="CC", scaling=1.0, prefix="lst"),
            dict(type="RSD", scaling=1.0, prefix="lst"),
        ]
    ),
    data_preprocessor=dict(  # TODO
        type="LSTDataPreprocessor",
        mean=None,
        std=None,
    ),
)