_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/datasets/heihe_lst/heihe_lst_x8-64-512.py",
    "../../_base_/schedules/schedule_20k.py",
]


scale = 8
# best configs!!!!
module = "MoCoLSKBlock" # # return attn*gui
model = dict(
    type="BaseEditModel",
    generator=dict(
        type="MoCoLSKNetABS",
        in_channels=1,
        gui_channels=10,
        num_feats=40,
        kernel_size=3,
        mocolsk_kernel_size=5,
        scale=scale,
        module=module,
        n_resblocks=4,
        num_stages=4,  # num of stages
        reduction=16,
        mlp_type='c',
        dmlp_num_layers = 3,
        cs_flag = ['S', 'C', 'S', 'C'], # # spatial or channel selection!
        norm_flag=1,  # 0: not norm; 1: z-score; 2: min-max norm; other number: assert error.
        norm_dict=dict(mean=282.51, std=14.59, min=243.53, max=320.85),  # 240m paras
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
