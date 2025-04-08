import torch
from pytorch_pretrained_vit.model import ViT
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision import datasets
import torchvision.transforms as transforms
import random
from sklearn.neighbors import LocalOutlierFactor

class splitMNISTdataset(torch.utils.data.Dataset):
    def __init__(self, train_test, trainType, class_label):
        self.trainType = trainType
        
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        self.dataset =  datasets.MNIST(root='data', train=train_test, download=True, transform=transform)
        idx = (np.array(self.dataset.targets) == class_label) |(np.array(self.dataset.targets) == class_label+1)
        self.dataset.data = self.dataset.data[idx]
        self.dataset.targets = self.dataset.targets[idx]
        #self.dataset.targets = [self.dataset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        class_list = [class_label, class_label+1]
        self.class_mapping = {c: i for i, c in enumerate(class_list)}
        self.class_0= np.where(np.array(self.dataset.targets) == class_label)[0].tolist()
        self.class_1= np.where(np.array(self.dataset.targets) == class_label+1)[0].tolist()
        

    def __len__(self):
        return self.dataset.data.shape[0]

    def __getitem__(self, index):
            img, raw_target = self.dataset[index]
            target = self.class_mapping[raw_target]
            if self.trainType == 'train':
                return img, target
            if self.trainType == 'train_1':
                if target == 0:
                    removeIndex = self.class_0.index(index)
                    index2 = random.choice(self.class_0[:removeIndex] + self.class_0[removeIndex+1:])
                else:
                    removeIndex = self.class_1.index(index)
                    index2 = random.choice(self.class_1[:removeIndex] + self.class_1[removeIndex+1:])
                img2, target2 = self.dataset[index2]
                return (img, img2), target


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT('B_16', pretrained=True)
        self.backbone.fc = torch.nn.Identity()

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, train_loader_1, device, args):
    model.eval()
    # Feature extraction from training samples
    feature_space = get_features(model, device, train_loader)
    with open(args.resultPath+'PreTrain_ViT_Features.pickle', 'wb') as f:
                pickle.dump(feature_space, f)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)

    # Update of center of training samples in embedding space
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.to(device)

    losses =[]
    bestLoss=1e20
    bestFeatureSpace = feature_space
    for epoch in range(args.epochs):
        # Training of Feature Extractor for one Epoch
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device)

        # Feature extraction from training samples
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        feature_space = get_features(model, device, train_loader)
        losses.append(running_loss)

        # Update of center of training samples in embedding space
        center = torch.FloatTensor(feature_space).mean(dim=0)
        center = F.normalize(center, dim=-1)
        center = center.to(device)

        # Saving of the model based on the minimum loss
        if (bestLoss >= running_loss):
            bestLoss = running_loss
            bestFeatureSpace = feature_space
            states = {'state_dict': model.state_dict()}
            torch.save(states, os.path.join(args.resultPath, 'ViT_Model.pth'))
            with open(args.resultPath+'FineTune_ViT_Features.pickle', 'wb') as f:
                pickle.dump(feature_space, f)

    # Plot of number of epochs and loss
    plt.figure()
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss')
    plt.plot(np.arange(0, args.epochs), losses[:], 'b')
    plt.legend(('Loss'), loc='upper right')
    plt.savefig(args.resultPath + 'ViT_Model.jpg')

    return bestFeatureSpace

def get_features(model, device, train_loader):

    # Extract features from training samples
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader):
            imgs = imgs.repeat(1, 3, 1, 1)
            imgs = imgs.to(device)
            features = model(imgs[:,0:3,:,:])
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    return train_feature_space

def run_epoch(model, train_loader, optimizer, center, device):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):
    # for (img1, img2) in tqdm(train_loader, desc='Train...'):
        img1 = img1.repeat(1, 3, 1, 1)
        img2 = img2.repeat(1, 3, 1, 1)
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1[:,0:3,:,:])
        out_2 = model(img2[:,0:3,:,:])
        out_1 = out_1 - center
        out_2 = out_2 - center

        center_loss = ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())
        loss = contrastive_loss(out_1, out_2) + center_loss

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading Dataloader
    trainset_1 = splitMNISTdataset(train_test=True, trainType= 'train_1', class_label=args.label)
    trainset_1_loader = torch.utils.data.DataLoader(trainset_1, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    trainset = splitMNISTdataset(train_test=True, trainType= 'train', class_label=args.label)
    print(torch.unique(trainset.dataset.targets))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    model = Model()
    model = model.to(device)
    # Training of Feature Extractor
    fineTunedFeatures  = train_model(model, train_loader, trainset_1_loader, device, args)

    # In-domain Model
    inDomainModel = LocalOutlierFactor(novelty=True).fit(fineTunedFeatures)
    with open(args.resultPath+'inDomainModel.pickle', 'wb') as f:
             pickle.dump(inDomainModel, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training ViT Feature Extractor')
    parser.add_argument('--dataset', default='MNIST', help='MNIST')
    parser.add_argument('--resultPath', default='Feature_Extractor/ViT/MNIST/Task_5/', help='')
    parser.add_argument('--method', default='ViT', help='')
    parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=8, type=int, help='Class label')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=64, type=int, help= '64')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['TORCH_HOME'] = '../TempData/models/'
    if not os.path.exists(args.resultPath):
        os.makedirs(args.resultPath)
    main(args)