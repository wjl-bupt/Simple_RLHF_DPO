#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted from 1.train_gen.ipynb — training script for the generator model.
"""
import torch
from common import Tokenizer, ModelGEN, generate


def get_batch_data(tokenizer, device, batch_size=64):
    data = [tokenizer.get_data(third_number=False) for _ in range(batch_size)]

    # get max length
    lens = max([len(i) for i in data])

    # padding short sequences using 'P' token
    input_ids = torch.full((len(data), lens),
                           tokenizer.encoder['P'],
                           device=device)

    for i, d in enumerate(data):
        input_ids[i, -len(d):] = torch.LongTensor(d)

    attention_mask = (input_ids != tokenizer.encoder['P']).long()

    return input_ids, attention_mask


def main():
    tokenizer = Tokenizer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_gen = ModelGEN(device, tokenizer = tokenizer)
    optimizer = torch.optim.Adam(model_gen.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.encoder['P'])

    for i in range(2000):
        input_ids, attention_mask = get_batch_data(tokenizer, device)

        # out shape is (batch_size, seq_len, vocab_size), our vocab size is small (only 22)
        out = model_gen(input_ids=input_ids, attention_mask=attention_mask)

        # shift for next-token prediction
        # out[:, :-1] corresponds to input_ids[:, 1:] shape is (batch_size * seq_len - 1, vocab_size)
        # input_ids[:, 1:] is the target (next token), shape is (batch_size * seq_len - 1)
        loss = criterion(out[:, :-1].reshape(-1, out.size(-1)),
                         input_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            gen = generate(model_gen, input_ids[0].unsqueeze(0), device, tokenizer = tokenizer)
            print(i, tokenizer.decode(gen[0].tolist()), loss.item())

    model_gen.to('cpu')
    torch.save(model_gen.state_dict(), 'gen.model')


if __name__ == '__main__':
    main()
