import torch

x = torch.rand(10,768)
y = torch.rand(10,768)
m = torch.norm(x, dim=1)
z = torch.sum(x*y, dim=1) / torch.norm(x, dim=1) / torch.norm(y, dim=1)
z1 = torch.sum(x*y, dim=1) / torch.norm(x, dim=1) / torch.norm(y, dim=1)
z2 = torch.sum(x*y, dim=1) / torch.norm(x, dim=1) / torch.norm(y, dim=1)
e = torch.exp(-z)

print(x)
print(y)
print(x*y)
print(z.size())
print(e)