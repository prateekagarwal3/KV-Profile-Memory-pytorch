import torch
import torch.nn as nn
import torch.nn.functional as F

class KVMemNN(nn.Module):
    def __init__(self, mem_len, mem_size, embd_size, vocab_size):
        super(KVMemNN, self).__init__()
        self.A1 = nn.Embedding(vocab_size, embd_size)
        self.A1_bn = nn.BatchNorm1d(mem_len)
        self.A2 = nn.Embedding(vocab_size, embd_size)
        self.A2_bn = nn.BatchNorm1d(mem_len)
        self.mem_len = mem_len
        self.mem_size = mem_size
        self.embd_size = embd_size
        self.vocab_size = vocab_size
        self.W = nn.Linear(self.embd_size, self.embd_size)
    
    def forward(self, q, persona, key, val, cands):
        m = persona.view(-1, self.mem_len) #(bs*mem_size, mem_len)
        m = self.A1(m)
        m = self.A1_bn(m)
        m = m.view(-1, self.mem_size, self.mem_len, self.embd_size)
        m = torch.sum(m, dim=2) #(bs, mem_size, embd_size)

        cands = self.A1(cands)
        cands = self.A1_bn(cands)
        cands = cands.view(-1, 20, self.mem_len, self.embd_size)
        cands = torch.sum(cands, dim=2) #(bs, 20, embd_size)

        q = self.A1(q)
        q = torch.sum(q, dim=1) #(bs, embd_size)

        c = persona.view(-1, self.mem_len) #(bs*mem_size, mem_len)
        c = self.A2(c)
        c = self.A2_bn(c)
        c = c.view(-1, self.mem_size, self.mem_len, self.embd_size)
        c = torch.sum(c, dim=2) #(bs, mem_size, embd_size)

        p = torch.bmm(m, q.unsqueeze(2)).squeeze(2)
        # print(p.size())
        p = F.softmax(p, -1).unsqueeze(1)  # (bs, 1, mem_size)
        o = torch.bmm(p, c).squeeze(1)             # use m as c, (bs, embd_size)
        q = o + q # (bs, embd_size)

        key = self.A2(key) # (bs, mem_size, mem_len, embd_size)
        # key = self.A2_bn(key)
        key = torch.sum(key, dim=2) # (bs, mem_size, embd_size)

        val = self.A2(val) # (bs, mem_size, mem_len, embd_size)
        # val = self.A2_bn(val)
        val = torch.sum(val, dim=2) # (bs, mem_size, embd_size)

        ph = torch.bmm(key, q.unsqueeze(2)).squeeze(2)
        ph = F.softmax(ph, -1).unsqueeze(1)  # (bs, 1, mem_size)
        o = torch.bmm(ph, val).squeeze(1)  #(bs, embd_size)

        q = o + q #(bs, embd_size)
        q = self.W(q) #(bs, embd_size)
        q = torch.bmm(cands, q.unsqueeze(2)).squeeze(2) #(bs, 20)
        print(F.softmax(q, dim=1))
        return F.softmax(q, dim=1)

