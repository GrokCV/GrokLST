train_pipeline = [
    dict(type="LoadGrokLSTMatFile", key="lr_lst", data_field_name="data"),
    dict(type="LoadGrokLSTMatFile", key="hr_lst", data_field_name="data"),
    dict(type="PackLSTInputs"),
]

# dataset settings
dataset_type = "GrokLSTDataset"
lst_data_root = r"data/groklst"
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
        ),
        filename_tmpl=dict(),
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type="Evaluator",
    metrics=[
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
