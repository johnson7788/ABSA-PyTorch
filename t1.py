import torch
#初始化一个embedding字典
embedding = torch.nn.Embedding(10, 3, padding_idx=0)
input =  torch.LongTensor([[0,2,0,5]])
print(input.shape)
#打印input 做embedding后的结果
print(embedding(input))