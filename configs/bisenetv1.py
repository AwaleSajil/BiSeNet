
cfg = dict(
    model_type='bisenetv1',
    num_aux_heads=2,
    lr_start=3e-4,
    weight_decay=5e-5,
    warmup_iters=4000,
    max_iter=80000,
    im_root='./datasets/coco',
    train_im_anns='./datasets/coco/train.txt',
    val_im_anns='./datasets/coco/val.txt',
    scales=[0.5, 1.5],
    cropsize=[512, 512],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
