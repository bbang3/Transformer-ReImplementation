import torch
import os
from torch.utils.data import Dataset
from tokenizers import CharBPETokenizer

class TranslationDataset(Dataset):
    def __init__(self, path, split, src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256) -> None:
        super().__init__()
        
        src_lang, tgt_lang = language_pair.split('-')

        full_path = os.path.join(path, f'{split}.{src_lang}')
        with open(full_path, 'r', encoding='utf-8') as f:
            sents = f.readlines()
        src_sents = [sent.rstrip() for sent in sents]

        full_path = os.path.join(path, f'{split}.{tgt_lang}')
        with open(full_path, 'r', encoding='utf-8') as f:
            sents = f.readlines()
        tgt_sents = [sent.rstrip() for sent in sents]
    
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]

        src_token = self.src_tokenizer.encode(src_sent).ids
        tgt_token = self.tgt_tokenizer.encode(tgt_sent).ids

        return {
            'input': torch.tensor(src_token),
            'output': torch.tensor(tgt_token)
        }
    
    def _init_tokenizer(self, tokenizer_path):
        tokenizer = CharBPETokenizer(tokenizer_path)
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(length=self.max_length)
        return tokenizer