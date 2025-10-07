#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 PPO 后训练脚本，风格与 `train_dpo.py` 保持一致（过程式、全局 device、中文注释）。

说明：
- 该脚本作为示例，展示如何用生成器模型（`gen.model`）做简单的 PPO 微调。
- 请替换 `reward_fn` 为真实的 reward 模型或评估函数。
"""

import copy
import argparse
import re

import torch

from common import Tokenizer, generate, ModelGEN


def get_batch_data(batch_size=64, tokenizer=Tokenizer()):
    def pad(data, lens):
        # padding short sequences using 'P' token
        input_ids = torch.full((len(data), lens),
                               tokenizer.encoder['P'],
                               device=device)

        # attention mask
        for i, d in enumerate(data):
            input_ids[i, -len(d):] = torch.LongTensor(d)

        attention_mask = (input_ids != tokenizer.encoder['P']).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    # preferred answer.
    choice = [tokenizer.get_data(third_number=False) for _ in range(batch_size)]

    # reject answer simply defined as empty answer.
    prompts = [i[:i.index(tokenizer.encoder['=']) + 1] for i in choice]

    # get max lengths
    lens = max([len(p) for p in prompts])

    return pad(prompts, lens)



def gen_and_logprob(model, tokenizer, prompt_ids):
    """
    对单个或批量 prompt 采样并计算生成部分的 log-prob sum（适用于 PPO）。
    
    Args:
        model: 当前策略模型 (ModelGEN)
        tokenizer: Tokenizer 对象
        prompt_ids: [B, T] 或 [T] LongTensor，左填充的 prompt
    Returns:
        gen_ids: [B, T+gen_len] 生成序列（包含 prompt + 新生成）
        gen_logprob_sum: [B] 每个样本生成部分的 log-prob sum
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)  # batch维度

    B, T = prompt_ids.shape
    pad_id = tokenizer.encoder['P']

    # Attention mask: 左填充 -> padding部分 mask=0
    attention_mask = (prompt_ids != pad_id).long()

    # 1️⃣ 采样生成
    gen_ids = generate(model, prompt_ids, device, tokenizer=tokenizer)  # [B, T+gen_len]
    gen_ids = gen_ids.to(device)

    # 2️⃣ Attention mask 对齐
    attention_mask_full = (gen_ids != pad_id).long()

    # 3️⃣ 模型前向
    out = model(input_ids=gen_ids, attention_mask=attention_mask_full)  # [B, L, vocab]
    out = out[:, :-1]  # 对齐
    labels = gen_ids[:, 1:]  # 预测下一步
    logp = torch.log_softmax(out, dim=2)

    # 4️⃣ gather 对应 token 的 log-prob
    idx = labels.clone().unsqueeze(2)
    idx[idx == pad_id] = 0  # padding位置置0
    token_logp = logp.gather(2, idx).squeeze(2)  # [B, L-1]

    # 5️⃣ 生成部分 mask
    gen_len = gen_ids.size(1) - T
    gen_mask = torch.zeros_like(token_logp)
    gen_mask[:, -gen_len:] = 1.0  # 只保留生成部分

    # 6️⃣ 生成部分 log-prob sum
    gen_logprob_sum = (token_logp * gen_mask).sum(dim=1)  # [B]

    if gen_logprob_sum.size(0) == 1:
        return gen_ids[0], gen_logprob_sum[0]
    return gen_ids, gen_logprob_sum


def reward_fn(prompt_ids, resp_ids, tokenizer):
    """占位 reward：请用真实 reward 评估器替换。

    当前示例简单按回复长度打分。
    """
    # 规则化奖励函数：
    # 根据 tokenizer.get_data(third_number=True) 的生成格式，prompt 包含 target 在 '=' 之前（格式 S<target>=...）
    # 奖励由两部分构成：
    # 1) 如果 response 中包含恰好三个数字（即使用三个数字的组合），给 +1.0
    # 2) 如果把 response 解析为数学表达式并计算后与 target 相等（允许小数误差），给 +5.0

    # decode
    if torch.is_tensor(prompt_ids):
        prompt_texts = [tokenizer.decode(prompt) for prompt in prompt_ids.tolist()]
    else:
        prompt_texts = [tokenizer.decode(prompt_ids)]
    if torch.is_tensor(resp_ids):
        resp_texts = [tokenizer.decode(resp) for resp in resp_ids.tolist()]
    else:
        resp_texts = [tokenizer.decode(resp_ids)]   
    
    def get_single_rew (prompt_text, resp_text):
        # 提取 target（prompt 中 S 与 = 之间的数值）
        m = re.search(r'S(-?\d+\.?\d*)=', prompt_text)
        if not m:
            print('Warning: cannot find target in prompt:', prompt_text)
            # 无法解析 target，返回 0
            return 0.0

        try:
            target = float(m.group(1))
        except Exception:
            return 0.0

        # 从响应中提取允许的字符，防止出现字母或其他符号
        expr = ''.join(ch for ch in resp_text if ch in '0123456789.+-*/()')

        # 提取 response 中的数字序列，用于判断是否使用了三个数字
        numbers = re.findall(r'-?\d+\.?\d*', expr)

        reward = 0.0
        if len(numbers) - 1 == 3:
            reward += 1.0
        else:
            reward -= 1.0

        # # 计算表达式的数值（尽量安全地 eval）
        # try:
        #     # eval 在这里仅处理数学表达式，由于 expr 仅包含数字和运算符，因此风险较低
        #     val = eval(expr)
        #     min_rew, max_rew = -5.0, 5.0
        #     reward += min(max_rew, max(min_rew, 5.0 - abs(val - target)))

        # except Exception:
        #     # 解析或计算失败 -> no correctness reward
        #     pass

        return float(reward)

    
    rewards = [get_single_rew(p, r) for p, r in zip(prompt_texts, resp_texts)]
    return rewards


def ppo_update(policy, old_logps, new_logps, advantages, clip_eps, optimizer):
    """PPO 裁剪目标并执行一次更新。"""
    ratios = torch.exp(new_logps - old_logps)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    args = parser.parse_args()

    global device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer()

    # 加载策略模型
    policy = ModelGEN(device, tokenizer = tokenizer)
    policy.load_state_dict(torch.load('gen.model'))
    policy.train()

    # 备份一个参考/行为策略（初始与 policy 相同）
    policy_ref = ModelGEN(device, tokenizer = tokenizer)
    policy_ref.load_state_dict(torch.load('gen.model'))

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # baseline 用于 advantage
    baseline = 0.0
    baseline_alpha = 0.95

    for i in range(args.steps):
        data = get_batch_data(batch_size=args.batch_size, tokenizer=tokenizer)
        prompts_input_ids = data['input_ids']
        prompts_attention_mask = data['attention_mask']
        
        # use reference model to get old logps
        with torch.no_grad():
            full_ids_old, old_logps = gen_and_logprob(policy_ref, tokenizer, prompts_input_ids)

        # use current policy to get new logps and samples for policy optimization
        full_ids_new, new_logps = gen_and_logprob(policy, tokenizer, prompts_input_ids)

        rewards = reward_fn(prompts_input_ids, full_ids_new, tokenizer)
        
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

        adv = rewards - baseline
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * rewards.mean().item()

        loss_val = ppo_update(policy, old_logps, new_logps, adv, args.clip_eps, optimizer)

        if i % 100 == 0:
            print(i, 'loss', loss_val, 'reward_mean', rewards.mean().item(), 'baseline', baseline)
            print('sample:', tokenizer.decode(full_ids_new[0].tolist()).replace('P', '').replace('S', '').replace('E', ''))

        # 每隔若干步更新行为策略
        if (i + 1) % 50 == 0:
            policy_ref = copy.deepcopy(policy).eval()

    policy.to('cpu')
    torch.save(policy.state_dict(), 'ppo.model')


if __name__ == '__main__':
    main()
