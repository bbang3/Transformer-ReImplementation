from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

def prepare_tokenizer(path, lang, is_target, vocab_size=37000, max_length=256):
    if is_target:
        max_length -= 1 # [BOS] 토큰을 추가하기 때문에 max_length를 1만큼 줄여줌

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace() # to train a subword BPE tokenizer, we need to first tokenize the corpus by whitespace

    trainer = BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], vocab_size=vocab_size) # speical token은 입력한 순서대로 id 부여
    
    file = os.path.join(path, f'train.{lang}')
    tokenizer.train([file], trainer)

    # tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=max_length) # max length보다 짧으면 padding
    # tokenizer.enable_truncation(max_length=max_length) # 길면 truncation (special token 제외)
    
    # tokenizer.post_processor = TemplateProcessing(
    # single="[BOS] $A [EOS]",
    # special_tokens=[
    #         ("[BOS]", tokenizer.token_to_id("[BOS]")),
    #         ("[EOS]", tokenizer.token_to_id("[EOS]")),
    #     ],
    # )


    return tokenizer

def load_tokenizer(path, lang):
    path = os.path.join(path, f'tokenizer_{lang}.json')
    tokenizer = Tokenizer.from_file(path)
    return tokenizer