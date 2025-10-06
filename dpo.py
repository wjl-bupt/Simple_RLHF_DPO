# -*- encoding: utf-8 -*-
'''
@File :dpo.py
@Created-Time :2025-10-06 10:33:49
@Author  :june
@Description   : Direct Preference Optimization (DPO) implementation
@Modified-Time : 2025-10-06 10:33:49
'''

import torch
import torch.nn as nn


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