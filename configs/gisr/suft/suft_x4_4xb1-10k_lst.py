_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/heihe_lst/heihe_lst_x4-128-512.py",
    "../../_base_/schedules/schedule_10k.py",
]
scale = 4
# model settings
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="SUFTNet",
        in_channels=10,
        num_feats=32,
        kernel_size=3,
        scale=scale,
        norm_flag=2,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=282.51, std=14.68, min=242.72, max=321.21),  # 120m paras
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
    data_preprocessor=dict(
        type="LSTDataPreprocessor",
        mean=None,
        std=None,
    ),
)
