import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from tokenizer import prepare_tokenizer
from dataset import TranslationDataset
from model.transformer import Transformer
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

src_tokenizer = prepare_tokenizer('./data', 'en', is_target=False)
tgt_tokenizer = prepare_tokenizer('./data', 'de', is_target=True)

train_dataset = TranslationDataset('./data', 'train', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TranslationDataset('./data', 'validation', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# print(train_dataset[0])
ex = train_dataset[0]
print(ex['src'].shape, ex['tgt'].shape)

model = Transformer(device=device)
# model = nn.DataParallel(Transformer(device=device))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("[PAD]"))

wandb.init(
    project="Transformer-ReImplementation",
)

train(model, optimizer, criterion, train_loader, val_loader, device, tgt_tokenizer, num_epochs=10)

wandb.finish()