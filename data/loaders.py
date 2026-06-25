import torchvision.transforms as T
from torch.utils.data import DataLoader
from .dataset import FlickrDataset, get_collate_fn

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size=224):
    train = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    eval_ = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return train, eval_


def build_loaders(train_df, val_df, test_df, vocab, image_dir, cfg,
                  tokenizer=None, model_type='lstm'):
    """
    cfg keys:
        BATCH_SIZE   (int)   default 128
        NUM_WORKERS  (int)   default 2
        IMAGE_SIZE   (int)   default 224
        USE_CACHE    (bool)  default True
        MODEL_TYPE   (str)   default 'lstm'  — 'lstm'/'gru'/'minilm'
        MAX_LEN      (int)   default 50      — chỉ dùng khi MODEL_TYPE='minilm'
    """
    batch_size   = cfg.get('BATCH_SIZE', 128)
    num_workers  = cfg.get('NUM_WORKERS', 2)
    image_size   = cfg.get('IMAGE_SIZE', 224)
    use_cache    = cfg.get('USE_CACHE', True)
    max_len  = cfg.get('MAX_LEN', 50)

    # pad_idx lấy đúng nguồn
    if model_type == 'minilm':
        pad_idx = tokenizer.pad_token_id
    else:
        pad_idx = vocab.word2idx.get('<pad>', 0)

    transform_train, transform_eval = get_transforms(image_size)

    # Shared dataset kwargs
    _ds_kwargs = dict(
        image_dir=image_dir,
        vocab=vocab,
        tokenizer=tokenizer,
        use_cache=use_cache,
        model_type=model_type,
        max_seq_len=max_len,
    )

    train_ds = FlickrDataset(df=train_df, transform=transform_train,
                             deterministic=False, **_ds_kwargs)
    val_ds   = FlickrDataset(df=val_df,   transform=transform_eval,
                             eval_all_caps=True,  **_ds_kwargs)
    test_ds  = FlickrDataset(df=test_df,  transform=transform_eval,
                             eval_all_caps=True,  **_ds_kwargs)

    # Collate fn đúng theo model_type
    _collate = get_collate_fn(model_type, pad_idx=pad_idx)

    _loader_kwargs = dict(
        collate_fn=_collate,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **_loader_kwargs)

    _print_summary(train_ds, val_ds, test_ds, train_loader, model_type)

    return train_loader, val_loader, test_loader


def _print_summary(train_ds, val_ds, test_ds, train_loader, model_type):
    batch = next(iter(train_loader))

    if model_type == 'minilm':
        images, input_ids, lengths, names, attention_mask = batch
        cap_shape  = tuple(input_ids.shape)
        mask_shape = f"  attn_mask   : {tuple(attention_mask.shape)}\n"
    else:
        images, captions, lengths, names = batch
        cap_shape  = tuple(captions.shape)
        mask_shape = ""

    print("=" * 45)
    print(f"  Model type  : {model_type}")
    print(f"  Train pairs : {len(train_ds):>6}")
    print(f"  Val pairs   : {len(val_ds):>6}")
    print(f"  Test pairs  : {len(test_ds):>6}")
    print("-" * 45)
    print(f"  images      : {tuple(images.shape)}")
    print(f"  captions    : {cap_shape}")
    print(mask_shape, end="")
    print(f"  lengths     : {tuple(lengths.shape)}")
    print(f"  image_names : {len(names)}")
    print("=" * 45)