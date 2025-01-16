import torch
from tqdm import tqdm
import evaluate
import json
import os


# Generate only one sentence
def _generate(model, data_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)

    labels = data_loader.dataset.tgt_sents
    predictions = []
    
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader), desc='Test') as pbar:
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
        with tqdm(data_loader, total=len(data_loader), desc='Test') as pbar:
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


def test(model, data_loader, device, tgt_tokenizer, output_path):

    # print(predictions, labels)
    labels = data_loader.dataset.tgt_sents
    predictions = generate(model, data_loader, device, tgt_tokenizer)

    metrics = evaluate.load("bleu")
    result = metrics.compute(predictions=predictions, references=labels)
    print(result)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, 'score.json'), 'w+', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_path, 'output.de'), 'w+', encoding='utf-8') as f:
        for sent in predictions:
            f.write(sent + '\n')

    return result