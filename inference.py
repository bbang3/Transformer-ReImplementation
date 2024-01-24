import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import evaluate
import json

from dataset import TranslationDataset
from model.transformer import Transformer
from tokenizer import load_tokenizer

# Generate only one sentence
def _generate(model, data_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)

    labels = data_loader.dataset.tgt_sents
    predictions = []
    
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader), desc=f'Test') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                bos_token_id = tgt_tokenizer.token_to_id("[BOS]")
                eos_token_id = tgt_tokenizer.token_to_id("[EOS]")
                    
                enc_inputs = batch['src']
                dec_inputs = torch.LongTensor([bos_token_id] * enc_inputs.shape[0]).unsqueeze(1).to(device) # (bs, 1)
                label = batch['tgt'] # (bs, seq_len)

                while dec_inputs.shape[-1] <= data_loader.dataset.max_length:
                    output = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)
                    last_token = output[:, -1, :].argmax(dim=-1) # (bs, )

                    dec_inputs = torch.cat([dec_inputs, last_token.unsqueeze(1)], dim=-1) # (bs, seq_len + 1)
                    if last_token[0].item() == eos_token_id:
                        break
                
                output_sent = tgt_tokenizer.decode(dec_inputs[0].tolist())
                predictions.append(output_sent)
                break # Generate only one sentence

    return predictions

def generate(model, data_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)

    labels = data_loader.dataset.tgt_sents
    predictions = []

    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader), desc=f'Test') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                bos_token_id = tgt_tokenizer.token_to_id("[BOS]")
                eos_token_id = tgt_tokenizer.token_to_id("[EOS]")
                    
                enc_inputs = batch['src']
                dec_inputs = torch.LongTensor([bos_token_id] * enc_inputs.shape[0]).unsqueeze(1).to(device) # (bs, 1)
                label = batch['tgt'] # (bs, seq_len)

                while dec_inputs.shape[-1] <= data_loader.dataset.max_length:
                    output = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)
                    last_token = output[:, -1, :].argmax(dim=-1) # (bs, )

                    dec_inputs = torch.cat([dec_inputs, last_token.unsqueeze(1)], dim=-1) # (bs, seq_len + 1)
                    if last_token[0].item() == eos_token_id:
                        break
                
                output_sent = tgt_tokenizer.decode(dec_inputs[0].tolist())
                predictions.append(output_sent)

    return predictions


def test(model, data_loader, metrics, device, tgt_tokenizer):

    # print(predictions, labels)
    labels = data_loader.dataset.tgt_sents
    predictions = generate(model, data_loader, device, tgt_tokenizer)

    result = metrics.compute(predictions=predictions, references=labels)
    print(result)

    with open('./output/output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    with open('./output/result.de', 'w', encoding='utf-8') as f:
        for sent in predictions:
            f.write(sent + '\n')

    return


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Transformer(device=device)
    model.load_state_dict(torch.load('./checkpoints/model_10.pt', map_location=device))

    src_tokenizer = load_tokenizer('./tokenizer', 'en')
    tgt_tokenizer = load_tokenizer('./tokenizer', 'de')

    test_dataset = TranslationDataset('./data', 'dev', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    bleu = evaluate.load("bleu")

    test(model, test_loader, bleu, device, tgt_tokenizer)