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

from common.common import Tokenizer, generate, ModelGEN


def get_prompts(batch_size=8, tokenizer=Tokenizer()):
    """生成一批 prompt（仿照 notebook 的行为：返回 question 部分直到 '=' 为止）。"""
    prompts = []
    for _ in range(batch_size):
        token = tokenizer.get_data(third_number=True)
        try:
            idx = token.index(tokenizer.encoder['=']) + 1
            prompt = token[:idx]
        except ValueError:
            prompt = token
        prompts.append(prompt)
    return prompts


def gen_and_logprob(model, tokenizer, prompt_ids):
    """对单个 prompt 采样并计算生成部分的 log-prob sum。

    返回 (full_ids_list, gen_logprob_sum_tensor)
    """
    # prompt -> tensor
    prompt_t = torch.LongTensor(prompt_ids).unsqueeze(0).to(device)

    # 采样（使用 common.generate）
    gen = generate(model, prompt_t, device)
    gen = gen.to(device)

    # attention mask
    pad_id = tokenizer.encoder['P']
    attention_mask = (gen != pad_id).long()

    # 模型前向，得到 logits -> log-prob
    out = model(input_ids=gen, attention_mask=attention_mask)
    out = out[:, :-1]
    labels = gen[:, 1:]
    logp = torch.log_softmax(out, dim=2)

    # gather 对应 token 的 log-prob
    idx = labels.unsqueeze(2).clone()
    idx[idx == -100] = 0
    token_logp = logp.gather(2, idx).squeeze(2)  # [1, L-1]

    # 生成部分长度
    gen_len = gen.size(1) - prompt_t.size(1)
    if gen_len <= 0:
        return gen[0].tolist(), torch.tensor(0.0, device=device)

    gen_token_logp = token_logp[0, -gen_len:]
    gen_logprob_sum = gen_token_logp.sum()

    return gen[0].tolist(), gen_logprob_sum


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
    prompt_text = tokenizer.decode(prompt_ids)
    resp_text = tokenizer.decode(resp_ids)

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
    if len(numbers) == 3:
        reward += 1.0
    else:
        reward -= 1.0

    # 计算表达式的数值（尽量安全地 eval）
    try:
        # eval 在这里仅处理数学表达式，由于 expr 仅包含数字和运算符，因此风险较低
        val = eval(expr)
        min_rew, max_rew = -5.0, 5.0
        reward += min(max_rew, max(min_rew, 5.0 - abs(val - target)))

    except Exception:
        # 解析或计算失败 -> no correctness reward
        pass

    return float(reward)


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
        prompts = get_prompts(batch_size=args.batch_size, tokenizer=tokenizer)

        old_logps = []
        new_logps = []
        rewards = []
        samples = []

        for p in prompts:
            # 行为策略采样并计算 logprob
            with torch.no_grad():
                full_ids_old, old_lp = gen_and_logprob(policy_ref, tokenizer, p)

            # 当前策略下的 logprob
            # with torch.no_grad():
            full_ids_new, new_lp = gen_and_logprob(policy, tokenizer, p)

            # 计算 reward（替换为你的打分器）
            resp_ids = full_ids_new[len(p):]
            r = reward_fn(p, resp_ids, tokenizer)

            samples.append((p, full_ids_new))
            old_logps.append(old_lp.detach())
            new_logps.append(new_lp)
            rewards.append(r)

        old_logps_t = torch.stack(old_logps).to(device)
        new_logps_t = torch.stack(new_logps).to(device)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        adv = rewards_t - baseline
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * rewards_t.mean().item()

        loss_val = ppo_update(policy, old_logps_t, new_logps_t, adv, args.clip_eps, optimizer)

        if i % 10 == 0:
            print(i, 'loss', loss_val, 'reward_mean', rewards_t.mean().item(), 'baseline', baseline)
            # 打印一个样例
            pids, full = samples[0]
            print('sample:', tokenizer.decode(full))

        # 每隔若干步更新行为策略
        if (i + 1) % 50 == 0:
            policy_ref = copy.deepcopy(policy).eval()

    policy.to('cpu')
    torch.save(policy, 'gen_ppo.model')


if __name__ == '__main__':
    main()
