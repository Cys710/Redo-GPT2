import torch
import torch.nn as nn
from torch.nn import functional as F

# 全局配置
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learnig_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# 读文件
def readFile():
    with open('input.txt','r',encoding='utf-8') as f:
        text = f.read()

    # print("length of dataset in charaters: ", len(text))

    return text

# 做一个映射(编码)
def makeMapping(text):
    # 字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # print("".join(chars))
    # print(vocab_size)
    
    # 映射
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return encode,decode,vocab_size
    # print(encode("hii there"))
    # print(decode(encode("hii there")))

# 训练集 验证集
def splitData(text,encode):
    data = torch.tensor(encode(text),dtype = torch.long)

    print(data.shape,data.dtype)
    # print(data[:100])

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data,val_data

def get_batch(data):
    # data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    
    x,y = x.to(device),y.to(device)
    return x,y

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v      # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4,n_embd//4)
        # self.ffwd = FeedFoward(n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd,n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)


    def forward(self,idx,targets = None):
        B, T = idx.shape  # 必须放在最前面！

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device = device)) # (T,C)
        x = tok_emb + pos_emb     # (B,T,C)
        # x = self.sa_head(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)     #(4,8,65) -> (32,65)
            targets = targets.view(B*T)     #(4,8,) -> (32,)
            # 交叉熵函数 (样本数 类别数)
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:,-block_size:]
            
            logits,loss = self(idx_cond)
            
            logits = logits[:,-1,:]
            # 概率数值转换
            probs = F.softmax(logits,dim = -1)
            # 随机采样
            idx_next = torch.multinomial(probs,num_samples=1)
            # 拼接
            idx = torch.cat((idx,idx_next),dim = 1)
        return idx

@torch.no_grad
def estimate_loss(train_data,val_data):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == 'train' else val_data
            x,y = get_batch(data)
            logits , loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



if __name__ == '__main__':
    text = readFile()
    encode,decode,vocab_size = makeMapping(text)
    train_data,val_data = splitData(text,encode)

    xb,yb = get_batch(train_data)
    print(xb.shape,yb.shape)

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learnig_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(train_data,val_data)
            print(f"step {iter}: train loss {losses['train']:.4f},val {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- 训练循环结束 ---
    context = torch.zeros((1,1),dtype = torch.long,device = device)
    print(decode(m.generate(idx = context,max_new_tokens=500)[0].tolist()))
