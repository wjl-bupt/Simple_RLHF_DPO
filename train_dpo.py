#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted from 2.train_dpo.ipynb — DPO training script.
"""
import torch
from common import Tokenizer, ModelGEN, generate
from transformers import LlamaModel


class DPO(object):
    def __init__(self, model, model_ref, beta=1.0):
        self.model_dpo = model
        self.model_ref = model_ref
        self.beta = beta

    def get_prob_log(self, model, choice, reject):
        b = choice['input_ids'].shape[0]

        # 合并两部分输入,同时计算以提高效率
        input_ids = torch.cat([choice['input_ids'], reject['input_ids']], dim=0)
        attention_mask = torch.cat([
            choice['attention_mask'], reject['attention_mask']], dim=0)
        label = torch.cat([choice['label'], reject['label']], dim=0)

        # [2b, seq, vocab]
        out = model(input_ids=input_ids, attention_mask=attention_mask)

        # 偏移以对齐
        label = label[:, 1:]
        out = out[:, :-1]

        # 使用 log_softmax 来直接得到每个 token 的 log-prob（数值更稳定）
        out_logprob = torch.log_softmax(out, dim=2)

        # 取预测到 label 的 log-prob
        index = label.clone().unsqueeze(2)
        # 为避免负索引，把 -100 暂时置为 0（随后用 mask 排除这些位置）
        index[index == -100] = 0
        prob = out_logprob.gather(2, index=index).squeeze(2)

        # 只取答案部分的 log-prob，并在时间维度求和得到每个序列的联合 log 概率
        mask = (label != -100)
        prob = (prob * mask).sum(1)  # 返回 [2*b]，前 b 为 choice，后 b 为 reject

        return prob
    
    def compute_loss(self, choice, reject):
        # 获取选择和拒绝的概率对数
        logp_dpo_out = self.get_prob_log(self.model_dpo, choice, reject)
        logp_choice = logp_dpo_out[:choice['input_ids'].shape[0]]
        logp_reject = logp_dpo_out[choice['input_ids'].shape[0]:]

        with torch.no_grad():
            logp_ref_out = self.get_prob_log(self.model_ref, choice, reject)
            logp_choice_ref = logp_ref_out[:choice['input_ids'].shape[0]]
            logp_reject_ref = logp_ref_out[choice['input_ids'].shape[0]:]

        # calculate dpo loss   
        # $$
        # \mathcal{L} = -\log \sigma\left(\beta\left(\log p_{\theta}(\text{choice}) - \log p_{\theta}(\text{reject}) - \log p_{\text{ref}}(\text{choice}) + \log p_{\text{ref}}(\text{reject})\right)\right)
        # $$
        # where $\sigma$ is the sigmoid function.
        # In code, we compute the mean loss over the batch.
        loss = -torch.mean(
            torch.log(
                torch.sigmoid(
                    self.beta * ((logp_choice - logp_reject) - (logp_choice_ref - logp_reject_ref))
                )
            )
        )

        return loss


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
