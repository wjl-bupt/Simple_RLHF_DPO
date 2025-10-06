# -*- encoding: utf-8 -*-
'''
@File :common.py
@Created-Time :2025-10-06 10:08:52
@Author  :june
@Description   :    
@Modified-Time : 2025-10-06 10:08:52
'''



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import math
from transformers import LlamaConfig, LlamaModel, AutoModelForCausalLM



class Tokenizer:
    """ usage :
    tokenizer = Tokenizer()
    tokenizer.decode(tokenizer.get_data(third_number=True))
    token = tokenizer.get_data(third_number=True)
    """

    def __init__(self):
        self.vocab = {
            'mark': list('PSEU'),
            'number': list('0123456789'),
            'symbol': list('+-*/'),
            'other': list('.:=_')
        }

        self.decoder = [j for i in self.vocab.values() for j in i]
        self.encoder = {j: i for i, j in enumerate(self.decoder)}

    def get_data(self, third_number):
        question = ''
        for i in range(2):
            question += '%.2f' % random.uniform(-100, 100)
            question += random.choice(self.vocab['symbol'])

        question = question[:-1]
        if third_number:
            question += '+%.2f' % random.uniform(-100, 100)

        try:
            answer = '%.2f' % eval(question)
        except:
            answer = '0.00'

        # exchange question and answer 
        question, answer = answer, question

        # using 'S' as start token, 'E' as end token, 'P' as padding token
        token = 'S' + question + '=' + answer + 'E'
        # convert to id sequence
        token = [self.encoder[i] for i in token]
        # return id sequence (corresponding to token)
        return token

    def decode(self, token):
        return ''.join([self.decoder[i] for i in token])




class ModelGEN(nn.Module):

    def __init__(self, device, tokenizer=Tokenizer()):
        super().__init__()
        

        self.config = LlamaConfig(hidden_size=64,
                                  intermediate_size=64,
                                  max_position_embeddings=128,
                                  num_attention_heads=4,
                                  num_hidden_layers=4,
                                  num_key_value_heads=4,
                                  vocab_size=len(tokenizer.decoder))

        self.feature = LlamaModel(self.config)
        self.fc_out = torch.nn.Linear(64, self.config.vocab_size, bias=False)

        self.to(device)
        self.train()

    def forward(self, input_ids, attention_mask):
        out = self.feature(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state

        return self.fc_out(out)


generater = None


def generate(model_gen, input_ids, device, tokenizer = Tokenizer()):
    global generater
    if not generater:
        #包装类,用于生成
        generater = AutoModelForCausalLM.from_config(model_gen.config)
        generater.model = model_gen.feature
        generater.lm_head = model_gen.fc_out
        generater.to(device)

    return generater.generate(input_ids=input_ids,
                              min_length=-1,
                              top_k=0.0,
                              top_p=1.0,
                              do_sample=True,
                              pad_token_id=tokenizer.encoder['P'],
                              max_new_tokens=35,
                              eos_token_id=tokenizer.encoder['E'])