import torch
import math

from model import depthnet, losses, data

# hyperparameter
batch_size = 32
learning_rate = 0.001
total_epoch = 4
report_rate = 20

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Datasets and loader
dataset = data.DepthDataset("/dl/data/nyu-depth/")
lengths = [int(math.floor(len(dataset) * 0.8)), int(math.ceil(len(dataset) * 0.2))]
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

# load model
model = depthnet.DepthNet().to(device)

# Loss and optimizer
criterion = losses.BerHuLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# learning rate decay
def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

# validation
def validate(model, test_loader):
    model.eval()
    with torch.no_grad():
        loss = 0.0
        for t_image, t_depth in test_loader:
            t_image = t_image.to(device)
            t_depth = t_depth.to(device)
            t_outputs = model(t_image)
            curr_loss = criterion(t_depth, t_outputs)
            loss += curr_loss.item()
        print("Validation Loss: {:.4f}"
              .format(loss/(len(test_loader) * batch_size)))
    model.train()


# train
total_step = len(train_dataset)
curr_lr = learning_rate
for epoch in range(total_epoch):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, (image, depth) in enumerate(train_loader):
        image = image.to(device)
        depth = depth.to(device)

        # forward pass
        outputs = model(image)
        loss = criterion(outputs, depth)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate loss
        running_loss += loss.item()
        epoch_loss += running_loss

        if (i + 1) % report_rate == 0:
            print("Epoch: [{}/{}] Step [{}/{}] Loss: {:.4f}"
                  .format((epoch+1), total_epoch, (i+1), total_step, (running_loss/batch_size)))
            running_loss = 0.0

    #Decay learning rate
    if (epoch + 1) % 5 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    # Report epoch loss
    print("Epoch: [{}/{}] Epoch Loss: {:.4f}\n"
          .format((epoch+1), total_epoch, (epoch_loss / (len(train_loader) * batch_size))))

    validate(model, test_loader)

# Save the model checkpoint
torch.save(model.state_dict(), 'depthnet.ckpt')