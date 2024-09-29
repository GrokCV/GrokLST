_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/heihe_lst/heihe_lst_x2-256-512.py",
    "../../_base_/schedules/schedule_10k.py",
]
scale = 2
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
        norm_dict=dict(mean=282.51, std=14.73, min=242.50, max=322.33),  # 60m norm paras
        stem_use_dmlp=False,
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
