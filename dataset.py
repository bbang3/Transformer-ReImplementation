import torch
import os
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.nn.functional import pad

class TranslationDataset(Dataset):
    def __init__(self, path, split, src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256) -> None:
        super().__init__()
        src_lang, tgt_lang = language_pair.split('-')

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    
        self.src_sents = self._read_sentences(path, split, src_lang)
        self.tgt_sents = self._read_sentences(path, split, tgt_lang)

        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]

        src_token = self.src_tokenizer.encode(src_sent).ids
        tgt_token = self.tgt_tokenizer.encode(tgt_sent).ids

        label = tgt_token[:self.max_length - 1] + [self.tgt_tokenizer.token_to_id('[EOS]')] # label은 [BOS] 토큰을 제외
        tgt_token = [self.tgt_tokenizer.token_to_id('[BOS]')] + tgt_token[:self.max_length-1] # decoder input은 [EOS] 토큰을 제외
        assert len(label) <= self.max_length and len(tgt_token) <= self.max_length

        pad_value = self.src_tokenizer.token_to_id('[PAD]')
        src_token = pad(torch.LongTensor(src_token), pad=(0, self.max_length - len(src_token)), mode='constant', value=pad_value)
        label = pad(torch.LongTensor(label), pad=(0, self.max_length - len(label)), mode='constant', value=pad_value)
        tgt_token = pad(torch.LongTensor(tgt_token), pad=(0, self.max_length - len(tgt_token)), mode='constant', value=pad_value)

        return {
            'src': src_token,
            'tgt': tgt_token,
            'label': label
        }
    
    def _read_sentences(self, path, split, lang):
        file_name = f'{split}.{lang}'
        full_path = os.path.join(path, file_name)
        with open(full_path, 'r', encoding='utf-8') as f:
            sents = f.readlines()
        sents = [sent.rstrip() for sent in sents]
        return sents