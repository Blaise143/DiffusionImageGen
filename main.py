import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loaders import CustomData
from models import NoiseScheduler

diffuser = NoiseScheduler(1000)
data = CustomData(path="data")[20]

stds = []
means = []
for i in range(1000):#[1,10,100,300,500, 999]:
    img = diffuser(data,i)
    # perm_img = img.permute(1,2,0).numpy()
    # plt.imshow(perm_img)
    # plt.show()
    means.append(torch.mean(img).float())
    stds.append(torch.std(img).float())

    # print(torch.mean(img), torch.std(img))

plt.plot(stds, label="std")
plt.plot(means, label="mean")
plt.legend()
plt.grid()
plt.show()

# out = diffuser(data, 1000)
# print(out)

# d = data[20].permute(1,2,0).numpy()
# plt.imshow(d)
# plt.show()
# print(d)
# print(len(data))
