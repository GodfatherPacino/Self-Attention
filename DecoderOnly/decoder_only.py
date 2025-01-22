import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data

# 模型参数
d_model = 512   # 词嵌入维度
d_ff = 2048     # 前馈网络隐藏层维度
d_k = d_v = 64  # Q、K、V的维度
n_layers = 6    # Decoder层数
n_heads = 8     # 注意力头数

# 示例数据
# 增加更多训练数据
sentences = [
    "I love machine learning",
    "Deep learning is interesting",
    "Transformer is powerful",
    "Neural networks are amazing",
    "AI helps people work better",
    "Language models can generate text",
    "Deep learning needs big data",
    "Machine learning solves problems",
    "AI transforms the future",
    "Models learn from examples",
    "Data science is growing fast",
    "Neural networks process information",
    "Learning algorithms improve daily",
    "Technology changes everything",
    "Computers understand language now"
]

# 调整模型参数，使其更适合小数据集
d_model = 128   # 降低维度
d_ff = 512      # 降低前馈网络维度
n_layers = 3    # 减少层数

# 构建词典
def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(sentences)
vocab_size = len(vocab)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, x):
        x += self.pos_table[:x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + residual)

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, inputs, self_attn_mask):
        outputs = self.self_attn(inputs, inputs, inputs, self_attn_mask)
        outputs = self.ff(outputs)
        return outputs

class DecoderOnly(nn.Module):
    def __init__(self):
        super(DecoderOnly, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len]
        outputs = self.embedding(inputs)
        outputs = self.pos_encoder(outputs)
        
        # 生成因果注意力掩码（上三角掩码）
        seq_len = inputs.size(1)
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(inputs.size(0), -1, -1)

        for layer in self.layers:
            outputs = layer(outputs, subsequent_mask)

        return self.projection(outputs)

    def generate(self, start_token, max_len):
        self.eval()
        with torch.no_grad():
            current_seq = torch.LongTensor([[start_token]])
            generated_tokens = set()
            
            for _ in range(max_len-1):
                logits = self.forward(current_seq)
                temperature = 0.6
                logits = logits[:, -1:] / temperature
                
                # 处理概率分布
                logits = logits.squeeze()
                # 设置最小值避免数值问题
                logits = torch.clamp(logits, min=-100, max=100)
                
                # 过滤已生成的token和特殊标记
                for token in generated_tokens:
                    logits[token] *= 0.3
                logits[vocab['<PAD>']] = -float('inf')
                logits[vocab['<BOS>']] = -float('inf')
                
                # 使用 softmax 获取概率分布
                probs = torch.softmax(logits, dim=-1)
                # 确保概率和为1且没有无效值
                probs = torch.nan_to_num(probs, 0.0)
                if probs.sum() == 0:
                    probs = torch.ones_like(probs) / probs.size(0)
                
                # 采样下一个token
                try:
                    next_token = torch.multinomial(probs, 1)
                except RuntimeError:
                    # 如果采样失败，选择概率最大的token
                    next_token = torch.argmax(probs).unsqueeze(0)
                
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_tokens.add(next_token.item())
                
                if next_token.item() == vocab['<EOS>']:
                    break
                
                # 控制生成长度
                if len(current_seq[0]) >= 6:
                    current_seq = torch.cat([current_seq, torch.LongTensor([[vocab['<EOS>']]])], dim=1)
                    break
                    
            return current_seq.squeeze()

# 训练相关代码
def prepare_data(sentences, vocab):
    max_len = max(len(s.split()) for s in sentences) + 2  # +2 for BOS and EOS
    data = []
    for sentence in sentences:
        tokens = [vocab['<BOS>']] + [vocab[w] for w in sentence.split()] + [vocab['<EOS>']]
        tokens = tokens + [vocab['<PAD>']] * (max_len - len(tokens))
        data.append(tokens)
    return torch.LongTensor(data)

def train_model(model, data, epochs=300):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 添加L2正则化
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=1)
    
    best_loss = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output.view(-1, vocab_size), data.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            # 早停机制
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    # 准备训练数据
    train_data = prepare_data(sentences, vocab)
    
    # 初始化模型
    model = DecoderOnly()
    
    # 训练模型
    train_model(model, train_data)
    
    # 生成文本示例
    generated = model.generate(vocab['<BOS>'], max_len=10)
    # 修改文本生成的处理方式
    generated_text = []
    for idx in generated:
        word = [k for k, v in vocab.items() if v == idx.item()]
        if word:
            generated_text.append(word[0])
    generated_text = ' '.join(generated_text)
    print("Generated text:", generated_text)
    
    # 多次测试生成效果
    print("\n生成多个样本：")
    for i in range(5):
        generated = model.generate(vocab['<BOS>'], max_len=10)
        generated_text = []
        for idx in generated:
            word = [k for k, v in vocab.items() if v == idx.item()]
            if word and word[0] not in ['<PAD>', '<BOS>', '<EOS>']:
                generated_text.append(word[0])
        generated_text = ' '.join(generated_text)
        print(f"Sample {i+1}: {generated_text}")