import torch
import torch.nn as nn

m = nn.Sigmoid()
bceloss = nn.BCELoss()
input = torch.randn(5)
target = torch.FloatTensor([1,0,1,0,0])
output = bceloss(m(input), target)
print(output)

celoss = nn.CrossEntropyLoss()
input = torch.FloatTensor([[0.1,0.2,0.3,0.1,0.3],[0.1,0.2,0.3,0.1,0.3]])
target = torch.LongTensor([2,3])
output = celoss(input, target)
print(output)