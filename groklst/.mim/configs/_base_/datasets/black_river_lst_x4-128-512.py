# black_river_lst3 下 cat (各种波段的引导信息拼接起来) 文件夹下每个通道的引导数据没有进行任何归一化，在此，我们给出相关归一化参数
# 10种引导信息拼接顺序为dem, deepblue, blue, green, red, vre, nir, ndmvi, ndvi, ndwi;
## 120m 每种guidance数据的归一化参数：
gui_norm_paras = dict(
    dem=dict(mean=2569.36, std=1254.31, min=905.87, max=5070.05),
    deepblue=dict(mean=0.17, std=0.05, min=0.00, max=0.71),
    blue=dict(mean=0.16, std=0.04, min=0.00, max=0.45),
    green=dict(mean=0.15, std=0.04, min=0.00, max=0.37),
    red=dict(mean=0.17, std=0.05, min=0.00, max=0.40),
    vre=dict(mean=0.20, std=0.05, min=0.00, max=0.61),
    nir=dict(mean=0.22, std=0.06, min=0.00, max=0.54),
    ndmvi=dict(mean=0.19, std=0.12, min=-0.90, max=0.93),
    ndvi=dict(mean=0.14, std=0.09, min=-0.44, max=0.69),
    ndwi=dict(mean=-0.19, std=0.11, min=-0.56, max=0.68),
)

# LST 120m 数据的归一化参数：
lst_norm_paras = dict(mean=282.51, std=14.68, min=242.72, max=321.21)

norm_flag = 1

train_pipeline = [
    dict(
        type="LoadMatFile", key="lr_lst", data_field_name="data"
    ),  # optional keys =  ['hr_lst', 'lr_lst', 'hr_guidance', 'lr_mask', 'hr_mask']
    dict(type="LoadMatFile", key="hr_lst", data_field_name="data"),
    dict(type="LoadMatFile", key="hr_guidance", data_field_name="data"),
    dict(type="NormalizeData", key="hr_guidance", norm_flag=norm_flag, norm_paras_dict=gui_norm_paras),
    dict(type="PackLSTInputs"),
]

# dataset settings
dataset_type = "BasicLSTDataset"
lst_data_root = r"data/black_river_lst3"
train_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file="split/trainval.txt",
        # metainfo=dict(dataset_type=dataset_type, task_name="lst_downscale"),
        data_root=lst_data_root,
        data_prefix=dict(
            lr_lst="120m/lst",
            hr_lst="30m/lst",
            hr_guidance="30m/cat",
        ),
        filename_tmpl=dict(),
        pipeline=train_pipeline,
    ),
)


# test config
val_pipeline = train_pipeline
val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="split/test.txt",
        # metainfo=dict(dataset_type=dataset_type, task_name="lst_downscale"),
        data_root=lst_data_root,
        data_prefix=dict(
            lr_lst="120m/lst",
            hr_lst="30m/lst",
            hr_guidance="30m/cat",
        ),
        filename_tmpl=dict(),
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type="Evaluator",
    metrics=[
        # * use mask!
        dict(type="RMSE", scaling=1.0, prefix="lst"),
        dict(type="LSTMAE", scaling=1.0, prefix="lst"),  # abs(pred-gt)
        dict(type="BIAS", scaling=1.0, prefix="lst"),  # pred-gt
        dict(type="CC", scaling=1.0, prefix="lst"),
        dict(type="RSD", scaling=1.0, prefix="lst"),
    ],
)


# test config
test_pipeline = train_pipeline
test_dataloader = val_dataloader

test_dataloader = [
    test_dataloader,
]

test_evaluator = val_evaluator

test_evaluator = [
    test_evaluator,
]
