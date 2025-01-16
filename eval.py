import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import evaluate as hf_evaluate
from inference import generate

def evaluate(model, criterion, val_loader, device, tgt_tokenizer):
    model.eval()
    model = model.to(device)

    metric = hf_evaluate.load("bleu")
    losses = []

    # Val loss
    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader), desc='Validation') as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                enc_inputs, dec_inputs, labels = batch['src'], batch['tgt'], batch['label'] # (bs, seq_len)
                outputs = model(enc_inputs, dec_inputs) # (bs, seq_len, dec_vocab_size)

                outputs = outputs.contiguous().view(-1, outputs.shape[-1]) # (bs * seq_len, dec_vocab_size)
                labels = labels.contiguous().view(-1) # (bs * seq_len)

                loss = criterion(outputs, labels)
                losses.append(loss.item())
 
                pbar.set_postfix({"val_loss": loss.item()})

    # Generate
    val_loader = DataLoader(val_loader.dataset, batch_size=1, shuffle=False)
    predictions = generate(model, val_loader, device, tgt_tokenizer)
    references = val_loader.dataset.tgt_sents[:len(predictions)]

    print("Prediction:", predictions[0])
    print("GT:", references[0])

    bleu = metric.compute(predictions=predictions, references=references)['bleu']
    
    return np.mean(losses), bleu