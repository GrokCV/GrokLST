train_pipeline = [
    dict(
        type="LoadHeiheLSTMatFile", key="lr_lst", data_field_name="data"
    ),  # optional keys =  ['hr_lst', 'lr_lst', 'hr_guidance', 'lr_mask', 'hr_mask']
    dict(type="LoadHeiheLSTMatFile", key="hr_lst", data_field_name="data"),
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
            lr_lst="240m/lst",
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
            lr_lst="240m/lst",
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
