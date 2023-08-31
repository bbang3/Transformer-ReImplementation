import torch
from tqdm import tqdm
import numpy as np
import wandb
import evaluate as hf_evaluate

from tokenizer import prepare_tokenizer

def train(model, optimizer, criterion, train_loader, val_loader, device, tgt_tokenizer, num_epochs=1):
    model.train()
    model = model.to(device)
    losses = []

    for epoch in range(1, num_epochs + 1):
        with tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                enc_inputs, dec_inputs, labels = batch['src'], batch['tgt'], batch['label'] # (bs, seq_len)
                outputs = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)

                # decoded = tgt_tokenizer.decode(outputs[0].argmax(dim=-1).tolist())
                # print(len(decoded), decoded)
                outputs = outputs.contiguous().view(-1, outputs.shape[-1]) # (bs * seq_len, dec_vocab_size)
                labels = labels.contiguous().view(-1) # (bs * seq_len)

                # print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": np.mean(losses)})

                wandb.log({"train_loss": np.mean(losses), "epoch": epoch})
        
        try:
            if epoch % 2 == 0:
                eval_loss, bleu = evaluate(model, criterion, val_loader, device, tgt_tokenizer)
                print(eval_loss, bleu)
                wandb.log({"eval_loss": eval_loss, "bleu": bleu})
        except Exception as e:
            print(e)
            wandb.log({"is_error": 1})

        torch.save(model.state_dict(), f'./checkpoints/model_{epoch}.pt')
        
    return losses


def evaluate(model, criterion, val_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)

    metric = hf_evaluate.load("bleu")
    losses = []

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader), desc=f'Validation') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                enc_inputs, dec_inputs, labels = batch['src'], batch['tgt'], batch['label'] # (bs, seq_len)
                outputs = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)

                outputs = outputs.contiguous().view(-1, outputs.shape[-1]) # (bs * seq_len, dec_vocab_size)
                labels = labels.contiguous().view(-1) # (bs * seq_len)

                loss = criterion(outputs, labels)
                losses.append(loss.item())
 
                pbar.set_postfix({"val_loss": np.mean(losses)})

        labels = val_loader.dataset.tgt_sents
        predictions = []
        with tqdm(val_loader, total=len(val_loader), desc=f'Validation_BLEU') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                for i in range(batch['src'].shape[0]):
                    bos_token_id = tgt_tokenizer.token_to_id("[BOS]")
                    eos_token_id = tgt_tokenizer.token_to_id("[EOS]")
                        
                    enc_inputs = batch['src'][i, :].unsqueeze(0)
                    dec_inputs = torch.LongTensor([bos_token_id] * enc_inputs.shape[0]).unsqueeze(1).to(device) # (bs, 1)

                    while True:
                        output = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)
                        last_token = output[:, -1, :].argmax(dim=-1) # (bs, )

                        dec_inputs = torch.cat([dec_inputs, last_token.unsqueeze(1)], dim=-1) # (bs, seq_len + 1)
                        if last_token[0].item() == eos_token_id or dec_inputs.shape[-1] >= val_loader.dataset.max_length:
                            break
                    
                    output_sent = tgt_tokenizer.decode(dec_inputs[0].tolist())
                    predictions.append(output_sent)
        
        labels = labels[:len(predictions)]
        bleu = metric.compute(predictions=predictions, references=labels)['bleu']
    
    return np.mean(losses), bleu