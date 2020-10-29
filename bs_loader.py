import bs_dataset as dataset
import torchvision.transforms as transforms
import torch

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_workers=16):
    # Create Transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    trainset = dataset.BelgiumTSC(
        root_dir=data_dir, train=True,  transform=transform)
    testset = dataset.BelgiumTSC(
        root_dir=data_dir, train=False,  transform=transform)

    # Load Datasets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader