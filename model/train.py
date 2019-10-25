import torch
import torch.nn as nn

import model
import data
import losses

# hyperparameter
batch_size = 32
learning_rate = 0.001

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Datasets and loader
dataset = data.DepthDataset()

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=False)

# load model
model = model.DepthNet().to(device)

# Loss and optimizer
criterion = losses.RMSLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
total_step = len(loader)
for i, (image, depth) in enumerate(loader):
    image = image.to(device)
    depth = depth.to(device)

    # forward pass
    outputs = model(image)
    loss = criterion(outputs, depth)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print("Step [{}/{}] Loss: {:.4f}"
              .format(i, total_step, loss.item()))
