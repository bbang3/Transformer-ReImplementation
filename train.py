import torch
from tqdm import tqdm
import numpy as np
import wandb

from tokenizer import prepare_tokenizer

def train(model, optimizer, criterion, train_loader, val_loader, device, tgt_tokenizer, num_epochs=10):
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

                wandb.log({"train_loss": np.mean(losses)})
        
        eval_loss = evaluate(model, criterion, val_loader, device, tgt_tokenizer)
        wandb.log({"eval_loss": eval_loss})

        torch.save(model.state_dict(), f'./checkpoints/model_{epoch}.pt')
        
    return losses


def evaluate(model, criterion, val_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)
    losses = []

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader), desc=f'Validation') as pbar:
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                enc_inputs, dec_inputs, labels = batch['src'], batch['tgt'], batch['label'] # (bs, seq_len)
                outputs = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)

                outputs = outputs.contiguous().view(-1, outputs.shape[-1]) # (bs * seq_len, dec_vocab_size)
                labels = labels.contiguous().view(-1) # (bs * seq_len)

                loss = criterion(outputs, labels)
                losses.append(loss.item())
 
                pbar.set_postfix({"val_loss": np.mean(losses)})
            
    
    return np.mean(losses)