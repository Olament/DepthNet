import torch

from model import depthnet, losses, data

# hyperparameter
batch_size = 32
learning_rate = 0.001
total_epoch = 20

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Datasets and loader
dataset = data.DepthDataset("data/")

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=False)

# load model
model = depthnet.DepthNet().to(device)

# Loss and optimizer
criterion = losses.RMSLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
total_step = len(loader)
for j in range(total_epoch):
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
            print("Epoch: [{}/{}] Step [{}/{}] Loss: {:.4f}"
                  .format((j+1), total_epoch, i, total_step, loss.item()))
