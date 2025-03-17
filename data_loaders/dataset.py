import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

class CustomData(Dataset):
    def __init__(self, path: str="../data", image_size: int = 256):
        super().__init__()
        self.image_paths = [os.path.join(path, fname) for fname in os.listdir(path)
                            if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        image = self.transform(image)
        return image




if __name__ =="__main__":
    data = CustomData()
    print(len(data))
    print(data[10].shape)
    d = data[10].permute(1,2,0).numpy()
    plt.imshow(d)
    plt.show()

