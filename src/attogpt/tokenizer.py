import regex as re
import json
import numpy as np

from attogpt.config import Config


class BPETokenizer:
    """BPE Tokenizer"""
    def __init__(
            self,
            config: Config,
            freq_thresh: int = 10,  # threshold for pair frequency
    ):
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.freq_thresh = freq_thresh

        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

        self.processing_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _merge_pair_in_cell(self, pair, cell):
        merged = ''.join(pair)
        new_cell = []
        i = 0
        while i < len(cell):
            if i < len(cell) - 1 and (cell[i], cell[i+1]) == pair:
                new_cell.append(merged)
                i += 2
            else:
                new_cell.append(cell[i])
                i += 1

        return new_cell

    def tokenize(self, dataset):
        """Use the BPE algorithm to tokenize the dataset"""

        self.merges = []

        # Process dataset
        vocabulary = sorted(list(set(dataset)))

        if len(vocabulary) > self.vocab_size:
            raise ValueError(
                f"Tokenization failed: corpus has {len(vocabulary)} unique characters, "
                f"which exceeds vocab_size={self.vocab_size}."
            )

        processed_dataset = re.findall(self.processing_pat, dataset)

        cells = [list(s) for s in processed_dataset]

        while len(vocabulary) < self.vocab_size:
            # Get highest frequency pair
            pairs = {}

            for cell in cells:
                for pair in zip(cell, cell[1:]):
                    pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            top_pair, freq = max(pairs.items(), key=lambda item: item[1])

            # Merge
            if freq < self.freq_thresh:
                break

            pair = tuple(top_pair)
            merged_pair = ''.join(pair)

            for idx, cell in enumerate(cells):
                if pair in zip(cell, cell[1:]):
                    cells[idx] = self._merge_pair_in_cell(pair, cell)

            vocabulary.append(merged_pair)
            self.merges.append(pair)
        
        if len(vocabulary) < self.vocab_size:
            raise ValueError(
                f"Tokenization failed: vocabulary has {len(vocabulary)} tokens, "
                f"but target vocab_size is {self.vocab_size}. "
                f"Try a larger corpus, lower freq_thresh, or reduce vocab_size."
            )

        self.token_to_id = {token: idx for idx, token in enumerate(vocabulary)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        tokenized_corpus_flattened = [token for cell in cells for token in cell]
        tokenized_corpus_ids = [self.token_to_id[token] for token in tokenized_corpus_flattened]

        return tokenized_corpus_ids

    def to_file(self, tokenized_corpus_ids, vocabulary_output_path, corpus_output_path):
        with open(vocabulary_output_path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'merges': self.merges,
            }, f)
        
        np.save(corpus_output_path, np.array(tokenized_corpus_ids, dtype=np.int16))

    def from_file(self, vocabulary_path):
        with open(vocabulary_path) as f:
            data = json.load(f)

        if len(data['token_to_id']) != self.vocab_size:
            raise ValueError(
                f"Tokenization failed: loaded vocabulary has {len(data['token_to_id'])} tokens, "
                f"expected {self.vocab_size}."
            )

        self.token_to_id = data['token_to_id']
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.merges = [tuple(merge) for merge in data['merges']]

    def encode(self, text):
        """Encode input text. (Excludes symbols not in the vocabulary)"""
        processed_text = re.findall(self.processing_pat, text)
        cells = [list(s) for s in processed_text]

        for pair in self.merges:
            for idx, cell in enumerate(cells):
                if pair in zip(cell, cell[1:]):
                    cells[idx] = self._merge_pair_in_cell(pair, cell)

        flat = [token for cell in cells for token in cell]
        return [self.token_to_id[token] for token in flat if token in self.token_to_id]

    def decode(self, ids):
        """Decode from ids to text"""
        return ''.join(self.id_to_token[i] for i in ids)
