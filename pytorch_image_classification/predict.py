import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms

val_dir = './data/val/'
data_transforms = {
    'predict': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

dataset = {'predict' : datasets.ImageFolder(val_dir, data_transforms['predict'])}
dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 128, shuffle=False, num_workers=8)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    model = torch.load('./save_model/model.pth')
    print(model)
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    since = time.time()
    for input, label in dataloader['predict']:
        input = input.to(device)
        output = model(input)
        output = output.to(device)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_1 / len(dataset))
    print("Top 5 err: ", 1 - correct_5 / len(dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))