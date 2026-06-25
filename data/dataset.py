import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class FlickrDataset(Dataset):
    """
    Hỗ trợ ba loại model: 'lstm', 'gru', 'minilm' (MiniLM-L6).

    - model_type='lstm'/'gru': dùng vocab.numericalize()  -> tensor int IDs
    - model_type='minilm'    : dùng HuggingFace tokenizer -> input_ids + attention_mask

    Index theo (image, caption) pairs:
    - Train (deterministic=False): tất cả pairs (5 captions/ảnh)
    - Val/Test (deterministic=True): caption đầu tiên mỗi ảnh
    - eval_all_caps=True: toàn bộ pairs, sort theo ảnh
    """

    def __init__(self, df, image_dir, vocab, tokenizer, transform=None, use_cache=True, deterministic=False, eval_all_caps=False, model_type='lstm', max_seq_len=64):
        self.df            = df.reset_index(drop=True)
        self.image_dir     = image_dir
        self.vocab         = vocab
        self.tokenizer     = tokenizer
        self.transform     = transform
        self.deterministic = deterministic
        self.eval_all_caps = eval_all_caps
        self.model_type    = model_type
        self.max_seq_len   = max_seq_len
        self.cache         = {}

        # ------------ Build pairs ------------
        if deterministic and not eval_all_caps:
            self.pairs = (
                df.groupby('image', sort=False)
                  .first()
                  .reset_index()[['image', 'caption']]
                  .values.tolist()
            )
        elif eval_all_caps:
            self.pairs = (
                df.sort_values('image')
                  [['image', 'caption']]
                  .values.tolist()
            )
        else:
            self.pairs = df[['image', 'caption']].values.tolist()

        # ------------ Cache ảnh ------------
        if use_cache:
            unique = list({p[0] for p in self.pairs})
            for name in tqdm(unique, desc='Caching images'):
                path = os.path.join(image_dir, name)
                if os.path.exists(path):
                    img = Image.open(path).convert('RGB')
                    self.cache[name] = transform(img) if transform else img

    def _load_image(self, image_name):
        if image_name in self.cache:
            image = self.cache[image_name]
            if not isinstance(image, torch.Tensor) and self.transform:
                image = self.transform(image)
        else:
            image = Image.open(
                os.path.join(self.image_dir, image_name)
            ).convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image

    def _tokenize_rnn(self, caption):
        """vocab -> 1-D int64 tensor"""
        return torch.tensor(self.vocab.numericalize(caption), dtype=torch.long)

    def _tokenize_minilm(self, caption):
        """HuggingFace tokenizer -> {'input_ids': Tensor, 'attention_mask': Tensor}"""
        enc = self.tokenizer(
            caption,
            max_length=self.max_seq_len,
            padding=False,                                      # padding sẽ làm ở collate_fn
            truncation=True,
            return_tensors='pt',
        )
        # squeeze batch dim (1, L) -> (L,)
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, caption = self.pairs[idx]
        image = self._load_image(image_name)

        if self.model_type in ('lstm', 'gru'):
            caption_data = self._tokenize_rnn(caption)
        else:
            caption_data = self._tokenize_minilm(caption)

        return image, caption_data, image_name


# ------------ Collate functions ------------
def collate_fn_rnn(batch, pad_idx):
    """Dùng cho LSTM / GRU — caption_data là 1-D int tensor"""
    images, captions, image_names = zip(*batch)

    images  = torch.stack(images)
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)

    max_len = lengths.max().item()
    padded  = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded, lengths, list(image_names)


def collate_fn_minilm(batch, pad_token_id):
    """Dùng cho MiniLM-L6 — caption_data là dict {input_ids, attention_mask}"""
    images, captions, image_names = zip(*batch)

    images  = torch.stack(images)
    lengths = torch.tensor(
        [c['input_ids'].size(0) for c in captions], dtype=torch.long
    )

    max_len = lengths.max().item()
    input_ids      = torch.full((len(captions), max_len), pad_token_id,  dtype=torch.long)
    attention_mask = torch.zeros((len(captions), max_len), dtype=torch.long)

    for i, cap in enumerate(captions):
        L = cap['input_ids'].size(0)
        input_ids[i, :L]      = cap['input_ids']
        attention_mask[i, :L] = cap['attention_mask']

    return images, input_ids, lengths, list(image_names), attention_mask

def get_collate_fn(model_type, pad_idx):
    """
    Factory trả về đúng collate_fn theo model_type
    """
    if model_type in ('lstm', 'gru'):
        return lambda b: collate_fn_rnn(b, pad_idx=pad_idx)
    else:
        return lambda b: collate_fn_minilm(b, pad_token_id=pad_idx)