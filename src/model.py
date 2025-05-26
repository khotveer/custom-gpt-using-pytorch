import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        
        # regularization
        self.attn_dropout = nn.Dropout(config['attn_pdrop'])
        self.resid_dropout = nn.Dropout(config['resid_pdrop'])
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd']),
            c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd']),
            act     = NewGELU(),
            dropout = nn.Dropout(config['resid_pdrop']),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.block_size = config['block_size']
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(config['embd_pdrop']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config['learning_rate'], betas=train_config['betas'])
        
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device

        b, t = idx.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)


        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        
        x = self.transformer.drop(tok_emb + pos_emb)
      
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

class Trainer:
    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)

        self.device = config['device']
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0


    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def validate(self, val_loader):
        model = self.model
        config = self.config
        model.eval()
        total_loss = 0
        count = 0
        val_loader = DataLoader(self.test_dataset, batch_size=config['batch_size'], shuffle=False)
        with torch.no_grad():
            for x, y in val_loader:
                
                x, y = x.to(self.device), y.to(self.device)
                _, loss = model(x, y)
                total_loss += loss.item()
                count += 1

        avg_val_loss = total_loss / count
        # print(f"[Validation] Iter {self.iter_num}: val loss = {avg_val_loss:.4f}")
        model.train()
        return avg_val_loss


    def run(self):
        model, config = self.model, self.config
       
        self.optimizer = model.configure_optimizers(config)
       
        train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=config['batch_size'])
     
        
        model.train()
        
        self.iter_num = 0

        self.iter_time = time.time()
        
        data_iter = iter(train_loader)
        
        while True:

            try:
                batch = next(data_iter)
                
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
               
            batch = [t.to(self.device) for t in batch]
            
            x, y = batch

            logits, self.loss = model(x, y)

            model.zero_grad(set_to_none=True)

            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            
            self.optimizer.step()
            
            self.trigger_callbacks('on_batch_end')
            
            self.iter_num += 1
            
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num % 100 == 0:
                avg_val_loss = self.validate(self.test_dataset)
                print(self.iter_num, f": loss: {self.loss.item()} val loss: {avg_val_loss:.4f}")

            # termination conditions
            if config['max_iters'] is not None and self.iter_num >= config['max_iters']:
                break
              # Optional: Run validation every 500 steps
            if self.iter_num % 100 == 0:
                self.validate(self.test_dataset)
