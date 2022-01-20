# 参数


# 类别
label_dim = 2

# 序列文件路径
# file_path = '/kaggle/input/cancerenhancers/cancerEnhancers_filtered.csv'
# file_path = "../data/variants_with_seq.csv"
# 训练数据文件
train_file_path = 'data/train_20001_onehot.csv'
# 验证数据文件
valid_file_path = 'data/valid_20001_onehot.csv'
# 测试数据文件
test_file_path = 'data/test_20001_onehot.csv'

# 序列的长度
seq_len = 20001
# pad的填充索引
pad_idx = 0

# embedding词典大小
embedding_num = 5
# embedding维度
embedding_dim = 10

# 前置CNN层的参数

pre_cnn_output_dim = 10
pre_cnn_output_seq_len = 120
pre_cnn_n_layers=4
pre_cnn_kernel_size=5

# Transformer参数

trans_input_dim = pre_cnn_output_dim
trans_input_seq_len = 120
# encoder层数
n_layers = 2
# feedfoward网络中隐藏层大小
hidden_dim = 512
# 注意力的头数
att_head_num = 5
# dropuout比例
trans_drop_prob = 0.5


# 后置CNN参数
output_dim = 1

# 学习率
learning_rate = 0.01

# batch大小 
batch_size = 100

# epoch数
epoch = 10

# 打印间隔
log_interval = 100