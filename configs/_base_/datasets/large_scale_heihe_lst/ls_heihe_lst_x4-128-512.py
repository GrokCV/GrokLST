# Large Scale HeiheLST 下 guidance (各种波段的引导信息拼接起来) 文件夹下每个通道的引导数据没有进行任何归一化，在此，我们给出相关归一化参数
# 13种引导信息拼接顺序为dem, deepblue, blue, green, red, vre, nir, ciex, ciey, kndvi, ndmvi, ndvi, ndwi;
## 30m 每种guidance数据的归一化参数：
gui_norm_paras = dict(
    dem=dict(mean=3043.67, std=1226.95, min=905.42, max=5082.07),
    deepblue=dict(mean=0.18, std=0.07, min=0.00, max=0.71),
    blue=dict(mean=0.17, std=0.06, min=0.00, max=0.50),
    green=dict(mean=0.15, std=0.04, min=0.00, max=0.37),
    red=dict(mean=0.17, std=0.05, min=0.00, max=0.40),
    vre=dict(mean=0.21, std=0.07, min=0.00, max=0.65),
    nir=dict(mean=0.23, std=0.07, min=0.00, max=0.60),
    ciex=dict(mean=0.34, std=0.02, min=0.24, max=0.42),
    ciey=dict(mean=0.32, std=0.02, min=0.25, max=0.38),
    kndvi=dict(mean=0.03, std=0.04, min=0.00, max=0.46),
    ndmvi=dict(mean=0.19, std=0.12, min=-0.98, max=0.99),
    ndvi=dict(mean=0.14, std=0.09, min=-0.62, max=0.71),
    ndwi=dict(mean=-0.20, std=0.11, min=-0.57, max=0.85),
)

# LST 120m 数据的归一化参数：
# lst_norm_paras = dict(mean=278.33, std=15.34, min=241.47, max=325.43)

norm_flag = 1

# bands_loss_rate_dict 波段丢失率字典：包括每个波段对应的下标以及丢失率
# 例如, 'dem': (0, 0.5) 意味着, dem 波段 对应着 guidance (B,H,W,C=10) 的 第 0 个通道, 
# 即 guidance[B,H,W,0], 且其波段丢失率为0.5
bands_loss_rate_dict = {'dem': 0.5, 'deepblue': 0.5, 'blue': 0.5, 'green': 0.5, 'red': 0.5, 'vre': 0.5, 'nir': 0.5, 'ciex': 0.5, 'ciey': 0.5, 'kndvi': 0.5, 'ndmvi': 0.5, 'ndvi': 0.5, 'ndwi': 0.5}

# ["dem", "deepblue", "blue", "green", "red", "vre", "nir", "ciex", "ciey", "kndvi", "ndmvi", "ndvi", "ndwi"]
hr_guidance_data_field_name = ["dem", "deepblue", "blue", "green", "red", "vre", "nir", "ndmvi", "ndvi", "ndwi"]

train_pipeline = [
    dict(
        type="LoadHeiheLSTMatFile", key="lr_lst", data_field_name="data"
    ),  # optional keys =  ['hr_lst', 'lr_lst', 'hr_guidance', 'lr_mask', 'hr_mask']
    dict(type="LoadHeiheLSTMatFile", key="hr_lst", data_field_name="data"),
    # when key="hr_guidance" in LoadHeiheLSTMatFile, note that len(data_field_name) is the number of the HR guidance!!! so you can select the desired bands to train a model through data_field_name.
    dict(type="LoadHeiheLSTMatFile", key="hr_guidance", data_field_name=hr_guidance_data_field_name),
    dict(type="RandomDropBands", key="hr_guidance", seed=0, bands_loss_rate_dict=bands_loss_rate_dict, 
         drop_band=True, with_zero_padding=True),
    dict(type="NormalizeHeiheLSTData", key="hr_guidance", norm_flag=norm_flag, norm_paras_dict=gui_norm_paras),
    # pad_mode: ["constant", "reflect", "edge", "symmetric"]
    dict(type="PadBands", key="hr_guidance", pad_mode="reflect", pad_value=0.0),
    dict(type="PackLSTInputs"),
]

# dataset settings
dataset_type = "BasicLSTDataset"
lst_data_root = r"data/large_scale_heihe_lst"
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
            hr_guidance="30m/guidance",
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
            hr_guidance="30m/guidance",
        ),
        filename_tmpl=dict(),
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type="Evaluator",
    metrics=[
        # * use mask!
        dict(type="RMSE", scaling=1.0, prefix="ls_heihe_lst"),
        dict(type="LSTMAE", scaling=1.0, prefix="ls_heihe_lst"),  # abs(pred-gt)
        dict(type="BIAS", scaling=1.0, prefix="ls_heihe_lst"),  # pred-gt
        dict(type="CC", scaling=1.0, prefix="ls_heihe_lst"),
        dict(type="RSD", scaling=1.0, prefix="ls_heihe_lst"),
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
