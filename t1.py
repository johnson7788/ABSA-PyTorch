import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# 输入维度[batch_size, seq_lengh] [3,4]
seq = torch.tensor([[1,2,0,0], [3,0,0,0], [4,5,6,0]])
#每个序列中的实际的长度
lens = [2, 1, 3]
packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
print(packed)
seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
#序列还原回来了
print(seq_unpacked)
print(lens_unpacked)
