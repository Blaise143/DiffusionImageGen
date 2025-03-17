import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loaders.dataset import CustomData
from models.ddpm import DenoisingModel

time_steps = 1000
image_size = 256
in_channels = 3
out_channels = 3
num_epochs = 20
learning_rate = 3e-4
batch_size = 32
model = DenoisingModel(time_steps, sample_size=image_size, in_channels=3, out_channels=3)

dataset = CustomData("../data")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model=model, dataloader=dataloader,
          batch_size = batch_size, num_epochs=num_epochs, learning_rate=learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_model_path = "../checkpoints/model.pth"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            t = torch.randint(0, time_steps, (batch_size,))
            noise = torch.randn_like(batch)
            noisy_batch = model.noise_scheduler(batch, t)
            predicted_noise = model.unet(noisy_batch, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(epoch_loss)
        avg_loss = epoch_loss / len(dataloader)


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")


if __name__ == "__main__":
    train(num_epochs=20)
    # for i in range(len(dataset)):
    #     print(dataset[i].shape)
