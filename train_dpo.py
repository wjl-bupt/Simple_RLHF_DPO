#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted from 2.train_dpo.ipynb — DPO training script.
"""
import torch
from common import Tokenizer, ModelGEN, generate
from transformers import LlamaModel
from dpo import DPO


def get_batch_data(batch_size=64, tokenizer=Tokenizer()):
    def pad(data, split, lens):
        # 做个白板 (pad)
        input_ids = torch.full((len(data), lens),
                               tokenizer.encoder['P'],
                               device=device)

        # 往白板里黏贴数据
        for i, d in enumerate(data):
            input_ids[i, :len(d)] = torch.LongTensor(d)

        attention_mask = (input_ids != tokenizer.encoder['P']).long()

        # 计算 label
        label = input_ids.clone()
        for l, s in zip(label, split):
            # 问题和 pad 的位置是 -100
            l[:s] = -100
            l[l == tokenizer.encoder['P']] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    # 正确的问答
    choice = [tokenizer.get_data(third_number=True) for _ in range(batch_size)]

    # 错误的回答简单地定义为空回答就可以了
    split = [i.index(tokenizer.encoder['=']) + 1 for i in choice]
    reject = [d[:s] for d, s in zip(choice, split)]
    reject = [i + [tokenizer.encoder['E']] for i in reject]

    # 求最大长度
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
        choice, reject = get_batch_data()

        # # calculate log prob from dpo model and reference model
        # prob_log = get_prob_log(model_dpo, choice, reject)
        # # Warning: reference model does not update, so no gradient needed
        # with torch.no_grad():
        #     prob_log_ref = get_prob_log(model_dpo_ref, choice, reject)

        # # calculate kl divergence, discrete space is easy to compute
        # kl = -0.1 * (prob_log - prob_log_ref)

        # # 以 kl 散度计算 loss
        # loss = (kl.sigmoid() + 1e-8).log().mean()
        
        loss = dpo_cls.compute_loss(choice, reject)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            question = tokenizer.get_data(third_number=True)
            question = question[:question.index(tokenizer.encoder['=']) + 1]
            question = torch.LongTensor(question).unsqueeze(0).to(device)

            gen = generate(model_dpo, question, device)
            print(i, tokenizer.decode(gen[0].tolist()))

    model_dpo.to('cpu')
    torch.save(model_dpo.state_dict(), 'dpo.model')



if __name__ == '__main__':
    main()
