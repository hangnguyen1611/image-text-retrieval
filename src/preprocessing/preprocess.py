import os
import numpy as np
import pandas as pd
import re

class DataSplitter:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def split_data(self, df_captions, image_id_col='image', ratios=(0.7, 0.15, 0.15)):
        """Chia dữ liệu theo tỉ lệ tuỳ chỉnh (mặc định 70/15/15)"""
        unique_ids = df_captions[image_id_col].unique()
        shuffled = self.rng.permutation(unique_ids)

        total = len(shuffled)
        n_train = int(total * ratios[0])
        n_val   = int(total * ratios[1])

        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train : n_train + n_val]
        test_ids  = shuffled[n_train + n_val:]

        train_df = df_captions[df_captions[image_id_col].isin(train_ids)].copy()
        val_df   = df_captions[df_captions[image_id_col].isin(val_ids)].copy()
        test_df  = df_captions[df_captions[image_id_col].isin(test_ids)].copy()

        return train_df, val_df, test_df

    def split_karpathy(self, df_captions, image_id_col='image',
                       n_train=6000, n_val=1000, n_test=1000):
        """
        Chia theo chuẩn Karpathy: 6000 / 1000 / 1000 ảnh
        """
        unique_ids = df_captions[image_id_col].unique()
        total = len(unique_ids)

        required = n_train + n_val + n_test
        assert total >= required, (
            f"Dataset chỉ có {total} ảnh, cần ít nhất {required}"
        )

        shuffled = self.rng.permutation(unique_ids)

        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train : n_train + n_val]
        test_ids  = shuffled[n_train + n_val : n_train + n_val + n_test]

        train_df = df_captions[df_captions[image_id_col].isin(train_ids)].copy()
        val_df   = df_captions[df_captions[image_id_col].isin(val_ids)].copy()
        test_df  = df_captions[df_captions[image_id_col].isin(test_ids)].copy()

        self._print_stats(train_df, val_df, test_df, image_id_col)
        return train_df, val_df, test_df

    def _print_stats(self, train_df, val_df, test_df, image_id_col):
        """In thống kê số ảnh và caption của từng tập"""
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            n_imgs = df[image_id_col].nunique()
            n_caps = len(df)
            print(f"  {name:5s}: {n_imgs:5d} ảnh | {n_caps:6d} captions")

    def save_splits(self, train_df, val_df, test_df, output_dir):
        """Lưu các file đã chia vào thư mục chỉ định"""
        os.makedirs(output_dir, exist_ok=True)

        splits = {
            'train_captions.csv': train_df,
            'val_captions.csv':   val_df,
            'test_captions.csv':  test_df,
        }

        for filename, df in splits.items():
            save_path = os.path.join(output_dir, filename)
            df.to_csv(save_path, index=False)

        print(f"--> [Split Done] Đã lưu các tập dữ liệu vào: {output_dir}")

class CaptionProcessor:
    def __init__(self):
        pass 

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_dataframe(self, df, caption_col='caption', image_col='image'):
        cleaned_rows = []
        for row in df.itertuples(index=False):
            img_name  = getattr(row, image_col)
            raw_cap   = getattr(row, caption_col)
            clean_cap = self.clean_text(raw_cap)  

            cleaned_rows.append({
                image_col:  img_name,
                caption_col: clean_cap
            })
        return pd.DataFrame(cleaned_rows)

    def save_to_csv(self, df, file_path):
        df.to_csv(file_path, index=False)
        print(f"--> [Saved] {file_path}")

    def load_from_csv(self, file_path):
        return pd.read_csv(file_path)