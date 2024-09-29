train_pipeline = [
    dict( type='LoadImageFromFile', # H W C
        key='img', # cloudy optical image
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict( type='LoadImageFromFile', # H W C
        key='gt', # clear optical image
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    # dict( type='LoadImageFromFile', # H W -> H W C
    #     key='sar_vv', # SAR image with VV mode.
    #     # color_type: candidates are color, grayscale, unchanged...
    #     # https://mmcv.readthedocs.io/zh-cn/latest/api/generated/mmcv.image.imread.html#mmcv.image.imread
    #     color_type='grayscale', 
    #     channel_order='rgb', # channel_order: candidates are 'bgr' and 'rgb'.
    #     imdecode_backend='cv2'),
    dict( type='LoadImageFromFile', # H W -> H W C
        key='sar_vh', # SAR image with VH mode.
        # color_type: candidates are color, grayscale, unchanged...
        # https://mmcv.readthedocs.io/zh-cn/latest/api/generated/mmcv.image.imread.html#mmcv.image.imread
        color_type='grayscale', 
        # channel_order='rgb', # channel_order: candidates are 'bgr' and 'rgb'.
        imdecode_backend='cv2'),
    # norm 
    dict(type="NormalizeOpticalSARData", key="sar_vv", norm_flag='zscore', mean=0, std=1),
    dict(type="NormalizeOpticalSARData", key="sar_vh", norm_flag='zscore', mean=0, std=1),
    dict(type="PackOpticalSARInputs"), # todo
]

# dataset settings
dataset_type = "OpticalSARDataset"
data_root = r"data/optical_sar_cloud_removal"
train_dataloader = dict(
    num_workers=0,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file="split/train.txt",
        data_root=data_root,
        data_prefix=dict(
            gt="train/opt_clear", # optical_clear 
            img="train/opt_cloudy", # optical_cloudy
            sar="train/SAR", # SAR
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
        ann_file="split/val.txt",
        data_root=data_root,
        data_prefix=dict(
            gt="train/opt_clear", # optical_clear 
            img="train/opt_cloudy", # optical_cloudy
            sar="train/SAR", # SAR
        ),
        filename_tmpl=dict(),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(
    type="Evaluator",
    metrics=[
        dict(type='SSIM'),
        dict(type="CC",),
        dict(type='MSE'),
        dict(type='PSNR'),
        dict(type='MAE'),
    ],
)

# test config
test_pipeline = train_pipeline
test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file="split/test.txt",
        data_root=data_root,
        data_prefix=dict(
            img="test/opt_cloudy", # optical_cloudy
            sar="test/SAR", # SAR
        ),
        filename_tmpl=dict(),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)


test_dataloader = [
    test_dataloader,
]

test_evaluator = val_evaluator

test_evaluator = [
    test_evaluator,
]
