import torch
from tqdm import tqdm
import numpy as np
import wandb

from eval import evaluate

def train(args, model, optimizer, criterion, train_loader, val_loader, device, tgt_tokenizer, num_epochs=1):
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
                pbar.set_postfix({"loss": loss.item()})
                
                wandb.log({"train_loss": loss.item(), "train_mean_loss": np.mean(losses), "epoch": epoch})
        try:
            eval_loss, bleu = evaluate(model, criterion, val_loader, device, tgt_tokenizer)
            print(eval_loss, bleu)
            wandb.log({"eval_loss": eval_loss, "bleu": bleu})
        except Exception as e:
            print(e)
            wandb.log({"is_error": 1})
        
        torch.save(model.state_dict(), f'./checkpoints/model_{args.run_name}_{epoch}.pt')
        
    return losses