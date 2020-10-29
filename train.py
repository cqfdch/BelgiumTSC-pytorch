from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import bs_loader
# from model import AlexNet
from model import Model

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, validate_loader = bs_loader.get_train_valid_loader('D:\\Networks\\cnn-ga-master\\data', batch_size=32, num_workers=0)

net = Model()
net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)
epoch = 30
save_path = './AlexNet.pth'
best_acc = 0.0
for epoch in range(epoch):
    # train
    net.train()
    running_loss = 0.0
    total = 0
    correct = 0
    show_step = 32
    for step, data in enumerate(tqdm(train_loader),0):
        images, labels = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().item()
        # print train process
        # rate = (step+1)/len(train_loader)
        # a = "*" * int(rate * 50)
        # b = "." * int((1 - rate) * 50)
        # print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        # print('Train-Epoch:%3d, %3d / %3d ,Loss: %.3f, Acc:%.3f'% (epoch+1, step+1, len(train_loader),running_loss/total, (correct/total)))
        if step % show_step == 0:
            print("Epoch [{}][{}/{}]:Loss:{:.3f},Acc:{:.3f}".format(epoch+1, step+1, len(train_loader),running_loss/total, (correct/total)))
    print()

    # validate
    net.eval()
    val_loss = 0.0
    total = 0
    correct = 0
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for _,val_data in enumerate(validate_loader,0):
            val_images, val_labels = val_data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = net(val_images)
            loss = loss_function(outputs, val_labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.detach(), 1)
            total += val_labels.size(0)
            correct += predicted.eq(val_labels.data).sum().item()
        if correct / total > best_acc:
            best_acc = correct / total
            # print('*'*100, self.best_acc)
            torch.save(net.state_dict(), save_path)
        print('Validate-Loss:%.3f, Acc:%.3f' % (val_loss / total, correct / total))


print('Finished Training')