_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/large_scale_heihe_lst/ls_heihe_lst_x4-128-512.py",
    "../../_base_/schedules/schedule_20k.py",
]
scale = 4
# model settings
# module=['CSExchangeBlock',]
module = "CSExchangeBlock"
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="DyFeXNet",
        in_channels=1,
        gui_channels=10,
        num_feats=32,
        kernel_size=3,
        scale=scale,
        module=module,
        norm_flag=1,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=278.33, std=15.34, min=241.47, max=325.43),  # 120m norm paras
        stem_use_dmlp=True,
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
        mean=None,  # do not norm
        std=None,
    ),
)
