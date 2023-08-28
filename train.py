from tqdm import tqdm
import numpy as np

from tokenizer import prepare_tokenizer

def train(model, optimizer, criterion, train_loader, device, tgt_tokenizer, num_epochs=10):
    model.train()
    losses = []

    for epoch in range(1, num_epochs + 1):
        with tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                enc_inputs, dec_inputs, labels = batch['src'], batch['tgt'], batch['label'] # (bs, seq_len)
                outputs = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)

                decoded = tgt_tokenizer.decode(outputs[0].argmax(dim=-1).tolist())
                print(len(decoded), decoded)
                outputs = outputs.contiguous().view(-1, outputs.shape[-1]) # (bs * seq_len, dec_vocab_size)
                labels = labels.contiguous().view(-1) # (bs * seq_len)

                # print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=np.mean(losses))
        
    return losses

