# %%
#coding=utf-8

'''
Enhancer Transformer的实现
'''

from numpy.core.fromnumeric import var
import torch
from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataloader import DataLoader
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, Subset
import pandas as pd



# %%
class VariantDataset(Dataset):
    def __init__(self, file_path) -> None:
        self.raw_data = pd.read_csv(file_path)
        self.encode_seq()

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, index):
        variant = self.raw_data.iloc[index]
        ref = torch.tensor(variant['encoded_ref'])
        alt = torch.tensor(variant['encoded_alt'])
        label = torch.tensor(variant['label'])
        return [ref, alt, label]
    
    def encode_seq(self):
        # 编码
        dict_ = {
            'A': 1,
            'T': 2,
            'C': 3,
            'G': 4,
            'N': 0
        }
        self.raw_data['encoded_ref'] = self.raw_data['seq_101_ref'].apply(lambda x: list(map(lambda i: dict_[i], list(x))))
        self.raw_data['encoded_alt'] = self.raw_data['seq_101_alt'].apply(lambda x: list(map(lambda i: dict_[i], list(x))))
                


# %%
class TransformerEmbedding(nn.Module):
    def __init__(self, embedding_num, embedding_dim, pad_idx, device) -> None:
        super().__init__()
        self.embedding = nn.Embedding(embedding_num, embedding_dim, pad_idx, device=device)

    def forward(self, x):
        x = self.embedding(x)
        return x

# %%
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        if d_model % 2 == 1:
            _2i = _2i[:-1]
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, d_model= x.size()
        # [batch_size = 128, seq_len = 30]

        return x + self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


# %%
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.view(batch_size, length, d_model)
        return tensor


# %%
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(3)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.view(batch_size, head, d_tensor, length)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


# %%
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


# %%
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# %%
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


# %%
class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# %%
class ClassificationLayer(nn.Module):
    def __init__(self, embedding_dim, seq_len, output_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 1)
        self.linear2 = nn.Linear(seq_len, output_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, _ = x.size() 
        x = F.relu(self.linear1(x))
        x = x.view(batch_size, -1)
        x = F.sigmoid(self.linear2(x))
        # x = self.sigmoid(x)
        return x


# %%
class VariantPathogenicityClassifier(nn.Module):
    def __init__(self, 
        pad_idx,
        pad_size,
        embedding_dim,
        att_head_num,
        hidden_dim,
        device,
        n_layers=3,
        drop_prob=0.1,
        output_dim=20) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.positionalEncoding = PositionalEncoding(
            d_model=embedding_dim,
            max_len=pad_size,
            device=device
        )
        self.encoder = Encoder(
            d_model=embedding_dim,
            ffn_hidden=hidden_dim,
            n_head=att_head_num,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device
        )
        self.classifier = ClassificationLayer(embedding_dim, pad_size, output_dim)
    
    def make_pad_mask(self, q, k, src_padding_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(src_padding_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(src_padding_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask
    
    def embedding(self, x, y, device):
        batch_size, length = x.size()
        x_ = torch.zeros(batch_size, length, 5).to(device)
        x = x.view(batch_size, -1, 1)
        x_.scatter_(2, x, 1)
        y_ = torch.zeros(batch_size, length, 5).to(device)
        y = y.view(batch_size, -1, 1)
        y_.scatter_(2, y, 1)
        return torch.cat((x_, y_), 2)
          
    def forward(self, x, y):
        src_mask = self.make_pad_mask(x, x, self.pad_idx)
        x = self.embedding(x, y, self.device)
        x = self.positionalEncoding(x)
        x = self.encoder(x, src_mask)
        x = self.classifier(x)
        return x


# %%
# 参数


# 类别
label_dim = 2
output_dim = 1

# 序列文件路径
# file_path = '/kaggle/input/cancerenhancers/cancerEnhancers_filtered.csv'
file_path = "../data/variants_with_seq.csv"

# pad的长度
pad_size = 101
# pad的填充索引
pad_idx = 0

# embedding词典大小
embedding_num = 5
# embedding维度
embedding_dim = 10

# encoder层数
n_layers = 3

# feedfoward网络中隐藏层大小
hidden_dim = 32
# # 注意力的头数
att_head_num = 1

# 学习率
learning_rate = 0.01

# batch大小 
batch_size = 200

# epoch数
epoch = 10

# 打印间隔
log_interval = 100


# %%
# dataset = VariantDataset(file_path)
# dataloader = DataLoader(dataset, batch_size=batch_size)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# %%
# trainset = VariantDataset('../data/train.csv')
trainset = VariantDataset('./data/train.csv')
# validset = VariantDataset('../data/valid.csv')
validset = VariantDataset('./data/valid.csv')
trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validLoader = DataLoader(validset, batch_size=batch_size, shuffle=True)

# %%
# dataset_size = len(dataset)
# valid_ratio = 0.2
# valid_size = int(valid_ratio*dataset_size)
# train_size = dataset_size - valid_size
# train_indices = [i for i in range(train_size)]
# valid_indices = [i for i in range(train_size, dataset_size)]
# trainset = Subset(dataset, train_indices)
# validset = Subset(dataset, valid_indices)
# trainLoader = DataLoader(trainset, batch_size=batch_size)
# validLoader = DataLoader(validset, batch_size=batch_size)


# %%
model = VariantPathogenicityClassifier(
    embedding_dim=embedding_dim,
    pad_idx=pad_idx,
    pad_size=pad_size,
    device=device,
    output_dim=output_dim,
    att_head_num=att_head_num,
    hidden_dim=hidden_dim,
    n_layers=n_layers
)

model=model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_func = BCELoss()


# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x, y, target) in enumerate(train_loader):
        x, y, target = x.to(device), y.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x, y)
        target = torch.unsqueeze(target, 1)
        # output=output.to(torch.float32)
        target=target.to(torch.float32)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# %%
def valid(model, device, validloader):
    model.eval()
    test_loss = 0
    correct = 0
    threshold = 0.5
    with torch.no_grad():
        for x, y, target in validloader:
            x, y, target = x.to(device), y.to(device), target.to(device)
            output = model(x, y)
            target = torch.unsqueeze(target, 1)
            target = target.to(torch.float32)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = (output > threshold).to(torch.float)
            pred = pred.reshape(1, -1)
            target = target.reshape(1, -1)
            correct += pred.eq(target).sum().item()
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validLoader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validLoader.dataset),
        100. * correct / (len(validLoader.dataset))))


# %%
for i in range(epoch):
    print(i)
    train(model, device, trainLoader, optimizer, i)
    valid(model, device, validLoader)

# %%
# valid(model, device, validLoader)

# %%
# att_heads = [1, 3]
# hidden_dims = [128, 256, 512, 1024]
# layers = [2, 4, 6, 8, 10]

# for head_num in att_heads:
#     for hidden_dim in hidden_dims:
#         for layer in layers:
#             print(head_num, hidden_dim, layer)
#             model = EnhancerClassifier(
#                 embedding_num = embedding_num,
#                 embedding_dim=embedding_dim,
#                 pad_idx=pad_idx,
#                 pad_size=pad_size,
#                 device=device,
#                 output_dim=label_dim,
#                 att_head_num=head_num,
#                 hidden_dim=hidden_dim,
#                 n_layers =layer
#             )
#             # %%
#             for i in range(epoch):
#                 print("-----epoch {}-----".format(i))
#                 train(model, device, trainLoader, optimizer, i)
#                 valid(model, device, validLoader)
        






