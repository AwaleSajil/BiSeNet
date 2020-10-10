
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    im_root='./datasets/coco',
    train_im_anns='./datasets/coco/train.txt',
    val_im_anns='./datasets/coco/val.txt',
    scales=[0.5, 1.5.],
    cropsize=[512, 512],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
