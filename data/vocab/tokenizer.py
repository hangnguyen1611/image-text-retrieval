import re
import json
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def _tokenize(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # bỏ dấu câu
        return text.split()

    def build_vocab(self, train_dataframe):
        counter = Counter()
        for caption in train_dataframe['caption'].astype(str):
            counter.update(self._tokenize(caption))

        current_idx = len(self.word2idx)
        for word, freq in sorted(counter.items()):  # sort để reproducible
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1

        print(f"--> [Vocab Done] Tổng số từ: {len(self.word2idx)}")

    def numericalize(self, text_input):
        if isinstance(text_input, list):
            return [self._num_single(t) for t in text_input]
        return self._num_single(text_input)

    def _num_single(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return []
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]   

    def get_padded_batch(self, sequences):
        max_len = max(len(s) for s in sequences)
        padded  = [s + [0] * (max_len - len(s)) for s in sequences]
        mask    = [[1 if i < len(s) else 0 
                    for i in range(max_len)] for s in sequences]
        return padded, mask

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}