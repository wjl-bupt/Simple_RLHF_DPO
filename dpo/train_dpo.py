#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted from 2.train_dpo.ipynb — DPO training script.
"""
import torch
from common.common import Tokenizer, ModelGEN, generate
from transformers import LlamaModel
from dpo.dpo import DPO


def get_batch_data(batch_size=64, tokenizer=Tokenizer()):
    def pad(data, split, lens):
        # padding short sequences using 'P' token
        input_ids = torch.full((len(data), lens),
                               tokenizer.encoder['P'],
                               device=device)

        # attention mask
        for i, d in enumerate(data):
            input_ids[i, :len(d)] = torch.LongTensor(d)

        attention_mask = (input_ids != tokenizer.encoder['P']).long()

        # caclulate label
        label = input_ids.clone()
        for l, s in zip(label, split):
            # question and padding token positions are -100
            l[:s] = -100
            l[l == tokenizer.encoder['P']] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    # preferred answer.
    choice = [tokenizer.get_data(third_number=True) for _ in range(batch_size)]

    # reject answer simply defined as empty answer.
    split = [i.index(tokenizer.encoder['=']) + 1 for i in choice]
    reject = [d[:s] for d, s in zip(choice, split)]
    reject = [i + [tokenizer.encoder['E']] for i in reject]

    # get max lengths
    lens = max([len(i) for i in choice])

    return pad(choice, split, lens), pad(reject, split, lens)


def main():
    # make sure common module has tokenizer available for functions/classes defined there
    tokenizer = Tokenizer()
    tokenizer = tokenizer
    
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device
    print(device)

    # (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. 
    #  Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. 
    #  Do it only if you got the file from a trusted source.
    # (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
    #     WeightsUnpickler error: Unsupported global: GLOBAL common.ModelGEN was not an allowed global by default. 
    #    Please use `torch.serialization.add_safe_globals([ModelGEN])` or 
    #   the `torch.serialization.safe_globals([common.ModelGEN])` context manager to allowlist this global if you trust this class/function.

    
    # load pretrained generator model (from gen model training)
    model_dpo = ModelGEN(device, tokenizer = tokenizer)
    model_dpo.load_state_dict(torch.load('gen.model'))
    
    
    model_dpo_ref = ModelGEN(device, tokenizer = tokenizer)
    model_dpo_ref.load_state_dict(torch.load('gen.model'))

    optimizer = torch.optim.Adam(model_dpo.parameters(),
                                 lr=1e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8)

    beta = 1.0
    dpo_cls = DPO(model_dpo, model_dpo_ref, beta=beta)

    
    for i in range(20_0000):
        # build a batch of choice and reject pairs
        choice, reject = get_batch_data()
        
        # calculate dpo loss. See dpo.py for details.
        loss = dpo_cls.compute_loss(choice, reject)
        
        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print training status every 100 steps
        if i % 100 == 0:
            question = tokenizer.get_data(third_number=True)
            question = question[:question.index(tokenizer.encoder['=']) + 1]
            question = torch.LongTensor(question).unsqueeze(0).to(device)

            gen = generate(model_dpo, question, device)
            print(i, tokenizer.decode(gen[0].tolist()))

    # save dpo model with the name dpo.model
    model_dpo.to('cpu')
    torch.save(model_dpo.state_dict(), 'dpo.model')



if __name__ == '__main__':
    main()
