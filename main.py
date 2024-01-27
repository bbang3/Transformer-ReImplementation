import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from tokenizer import prepare_tokenizer, load_tokenizer
from dataset import TranslationDataset
from model.transformer import Transformer
from train import train
from eval import evaluate
from inference import test

import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Transformer argument description', prefix_chars='--'  )

    parser.add_argument('--mode', type=str, default='train', help='train, eval, or test')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer path')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device type')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--run_name', type=str, default='From scratch', help='run name')

    args = parser.parse_args()

    return args

# python main.py --mode train --checkpoint ./checkpoints/model_from_epoch_2_3.pt --run_name from_epoch_3 --tokenizer_path ./tokenizer --lr 1e-4
if __name__ == "__main__":
    args = parse_argument()

    device = torch.device(args.device)
    print("Device: ", device)

    if args.tokenizer_path is None:
        src_tokenizer = prepare_tokenizer(args.data_path, 'en', is_target=False)
        tgt_tokenizer = prepare_tokenizer(args.data_path, 'de', is_target=True)
    else:
        print("Tokenizer is loaded from", args.tokenizer_path)
        src_tokenizer = load_tokenizer(args.tokenizer_path, 'en')
        tgt_tokenizer = load_tokenizer(args.tokenizer_path, 'de')

    train_dataset = TranslationDataset(args.data_path, 'train', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TranslationDataset(args.data_path, 'validation', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TranslationDataset(args.data_path, 'validation', src_tokenizer, tgt_tokenizer, language_pair='en-de', max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print(train_dataset[0])
    ex = train_dataset[0]
    print(ex['src'].shape, ex['tgt'].shape)

    model = Transformer(device=device)
    # model = nn.DataParallel(Transformer(device=device))
    if args.checkpoint is not None:
        print("Checkpoint is loaded from", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Learning rate: ", args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("[PAD]"))

    if args.mode == 'train':
        wandb.init(project="Transformer-ReImplementation", name=args.run_name)
        train(args, model, optimizer, criterion, train_loader, val_loader, device, tgt_tokenizer, num_epochs=args.num_epochs)
        wandb.finish()
    elif args.mode == 'eval':
        eval_loss, bleu = evaluate(model, criterion, val_loader, device, tgt_tokenizer)
        print(eval_loss, bleu)
    elif args.mode == 'test':
        test(model, test_loader, device, tgt_tokenizer, args.output_path)
    else:
        raise NotImplementedError("Please choose mode from train, eval, or test")