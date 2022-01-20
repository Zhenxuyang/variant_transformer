import imp
import copy
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataReader(Dataset):
    def __init__(self, file_path, embedding_fn) -> None:
        self.raw_data = pd.read_csv(file_path)
        self.embedding_fn = embedding_fn
        # self.encode_seq(seq_len, embedding_fn)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, index):
        variant = self.raw_data.iloc[index]
        ref = variant['Ref']
        alt = variant['Alt']
        seq = variant['seq']
        ref_embedded, alt_embedded= self.embedding_fn(ref, alt, seq)
        ref_embedded = torch.tensor(ref_embedded)
        alt_embedded = torch.tensor(alt_embedded)
        label = torch.tensor(variant['label'])
        return [ref_embedded, alt_embedded, label]
    
    # def make_pad_mask(self, q, k, src_padding_idx):
    #     len_q, len_k = q.size(1), k.size(1)

    #     # batch_size x 1 x 1 x len_k
    #     k = k.ne(src_padding_idx).unsqueeze(1).unsqueeze(2)
    #     # batch_size x 1 x len_q x len_k
    #     k = k.repeat(1, 1, len_q, 1)

    #     # batch_size x 1 x len_q x 1
    #     q = q.ne(src_padding_idx).unsqueeze(1).unsqueeze(3)
    #     # batch_size x 1 x len_q x len_k
    #     q = q.repeat(1, 1, 1, len_k)

    #     mask = k & q
    #     return mask
    
class Data():

    def __init__(self, data_file_path, seq_len) -> None:
        self.data_file_path = data_file_path
        self.seq_len = seq_len

    def read_data(self):
        dataset = DataReader(self.data_file_path, self.embedding)
        return dataset

    def embedding(self, ref, alt, seq) -> None:
        dict_ = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]
        }
        def encode(n):
            if n in dict_.keys():
                return dict_[n]
            else:
                return dict_['N']
        ref_embedded =  list(map(encode, list(seq)))
        alt_embedded = copy.deepcopy(ref_embedded)
        alt_embedded[len(seq)//2] = encode(alt)
        return ref_embedded, alt_embedded

    def split_train_valid_test(self) -> None:
        pass

class Pipeline():
    
    def __init__(self, config_path) -> None:
        self.params = imp.load_source('params', config_path)

    def load_data(self) -> None:
        train_data = Data(self.params.train_file_path, self.params.seq_len)
        valid_data = Data(self.params.valid_file_path, self.params.seq_len)
        train_dataset = train_data.read_data()
        valid_dataset = valid_data.read_data()
        return train_dataset, valid_dataset

    def run(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def predict(self) -> None:
        pass

pipeline = Pipeline('model/params.py')
d1, d2 = pipeline.load_data()