import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import math
import numpy as np
import argparse
import pickle
from pytorch_pretrained_vit.model import ViT
import torch.nn.functional as F
import os
from torchvision import datasets
from agents.default_custom import NormalNN
import models
from types import MethodType

class splitMNISTdataset(torch.utils.data.Dataset):
    def __init__(self, start_label, end_label, transform):
        self.dataset =  datasets.MNIST(root='data', train=False, download=True, transform=transform)
        target_labels = range(start_label, end_label+1)
        idx = np.isin(np.array(self.dataset.targets), target_labels)
        self.dataset.data = self.dataset.data[idx]
        self.dataset.targets = self.dataset.targets[idx]
        #self.class_mapping = {c: i for i, c in enumerate(target_labels)}

    def __len__(self):
        return self.dataset.data.shape[0]

    def __getitem__(self, index):
            img, raw_target = self.dataset[index]
            target = 0 if raw_target % 2 == 0 else 1
            return index, img, target

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT('B_16', pretrained=True)
        self.backbone.fc = torch.nn.Identity()

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def create_model(args, model_weights):
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[args.model_type].__dict__[args.model_name]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        out_dim = {'All': 2}
        for task,out_dim in out_dim.items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if model_weights is not None:
            print('=> Load model weights:')
            model_state = torch.load(model_weights,
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4,6'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32
    transform1 = transforms.Compose([
         transforms.Pad(2, fill=0, padding_mode='constant'),
         transforms.ToTensor(),
         normalize,
        ])
    
    transform2 = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    # Loading Dataloader
    testset1 = splitMNISTdataset(start_label=args.start_label, end_label=args.end_label, transform=transform1)
    print(torch.unique(testset1.dataset.targets))
    test_loader1 = torch.utils.data.DataLoader(testset1, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    testset2 = splitMNISTdataset(start_label=args.start_label, end_label=args.end_label, transform=transform2)
    print(torch.unique(testset2.dataset.targets))
    test_loader2 = torch.utils.data.DataLoader(testset2, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    # Load classifier models from Task1
    
    classifier1 = create_model(args, args.model_weights1)
    classifier1.eval()
    ViTModel1 = FeatureExtractor()
    weights = torch.load(args.taskFolder1+'ViT_Model.pth')
    ViTModel1.load_state_dict(weights['state_dict'])
    ViTModel1 = ViTModel1.to(device)
    ViTModel1.eval()
    with open(args.taskFolder1+'inDomainModel.pickle',"rb") as f:
        inDomainModel1 = pickle.load(f)

    classifier2 = create_model(args, args.model_weights2)
    classifier2.eval()
    ViTModel2 = FeatureExtractor()
    weights = torch.load(args.taskFolder2 + 'ViT_Model.pth')
    ViTModel2.load_state_dict(weights['state_dict'])
    ViTModel2 = ViTModel2.to(device)
    ViTModel2.eval()
    with open(args.taskFolder2 + 'inDomainModel.pickle', "rb") as f:
        inDomainModel2 = pickle.load(f)

    classifier3 = create_model(args, args.model_weights3)
    classifier3.eval()
    ViTModel3 = FeatureExtractor()
    weights = torch.load(args.taskFolder3+'ViT_Model.pth')
    ViTModel3.load_state_dict(weights['state_dict'])
    ViTModel3 = ViTModel3.to(device)
    ViTModel3.eval()
    with open(args.taskFolder3+'inDomainModel.pickle',"rb") as f:
        inDomainModel3 = pickle.load(f)

    classifier4 = create_model(args, args.model_weights4)
    classifier4.eval()
    ViTModel4 = FeatureExtractor()
    weights = torch.load(args.taskFolder4+'ViT_Model.pth')
    ViTModel4.load_state_dict(weights['state_dict'])
    ViTModel4 = ViTModel3.to(device)
    ViTModel4.eval()
    with open(args.taskFolder4+'inDomainModel.pickle',"rb") as f:
        inDomainModel4 = pickle.load(f)

    classifier5 = create_model(args, args.model_weights5)
    classifier5.eval()
    ViTModel5 = FeatureExtractor()
    weights = torch.load(args.taskFolder5+'ViT_Model.pth')
    ViTModel5.load_state_dict(weights['state_dict'])
    ViTModel5 = ViTModel5.to(device)
    ViTModel5.eval()
    with open(args.taskFolder5+'inDomainModel.pickle',"rb") as f:
        inDomainModel5 = pickle.load(f)


    total_samples = 0
    correct = 0
    for (index1, data1, targets1), (index2, data2, targets2) in zip(test_loader1, test_loader2):
        total_samples += data1.shape[0]
        assert (index1 == index2).all(), "Labels do not match!"
        assert (targets1 == targets2).all(), "Labels do not match!"
        out1 = classifier1(data1)
        out1 = out1[list(out1.keys())[0]]
        prediction1 = F.softmax(out1, dim=1)

        out2 = classifier2(data1)
        out2 = out2[list(out2.keys())[0]]
        prediction2 = F.softmax(out2, dim=1)
        
        out3 = classifier3(data1)
        out3 = out3[list(out3.keys())[0]]
        prediction3 = F.softmax(out3, dim=1)
        
        out4 = classifier4(data1)
        out4 = out4[list(out4.keys())[0]]
        prediction4 = F.softmax(out4, dim=1)

        out5 = classifier5(data1)
        out5 = out5[list(out5.keys())[0]]
        prediction5 = F.softmax(out5, dim=1)

        data2 = data2.repeat(1, 3, 1, 1).to(device)
        testFeatures1 = ViTModel1(data2)
        testFeatures2 = ViTModel2(data2)
        testFeatures3 = ViTModel3(data2)
        testFeatures4 = ViTModel4(data2)
        testFeatures5 = ViTModel5(data2)
        # Weight estimation from In-domain Model
        distanceBaseline = inDomainModel1.score_samples(testFeatures1.detach().cpu().numpy())
        distanceAddTrain = inDomainModel2.score_samples(testFeatures2.detach().cpu().numpy())
        distanceAddTrain2 = inDomainModel3.score_samples(testFeatures3.detach().cpu().numpy())
        distanceAddTrain3 = inDomainModel4.score_samples(testFeatures4.detach().cpu().numpy())
        distanceAddTrain4 = inDomainModel5.score_samples(testFeatures5.detach().cpu().numpy())
        # Stack distances into a tensor of shape [64, 2]
        distances = torch.tensor(np.stack((distanceBaseline, distanceAddTrain, distanceAddTrain2, distanceAddTrain3, distanceAddTrain4), axis=1))
        # Apply softmax along dimension 1 to get weights per sample
        weights = F.softmax(distances, dim=1)  # Output shape will be [batch_size, 2]
        # Fusion of scores
        score = weights[:, 0].unsqueeze(1) * prediction1 \
            + weights[:, 1].unsqueeze(1) * prediction2 \
                + weights[:, 2].unsqueeze(1) * prediction3 \
                + weights[:, 3].unsqueeze(1) * prediction4 \
                + weights[:, 4].unsqueeze(1) * prediction5
        pred = torch.max(score, dim=1)[1]
        correct += torch.sum(pred == targets1).int().item()
    print("Accuracy:" + str(correct/total_samples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training ViT Feature Extractor')
    parser.add_argument('--dataset', default='MNIST', help='MNIST')
    parser.add_argument('--start_label', default=8, type=int, help='Class label')
    parser.add_argument('--end_label', default=9, type=int, help='Class label')
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP400', help="The name of actual model for the backbone")
    parser.add_argument('--batch_size', default=64, type=int, help= '64')
    parser.add_argument('--model_weights1', type=str, 
                        default='/research/iprobe-paldebas/Research_Work/Continuous_Learning_MNIST/data/MNIST/Task_1/Task_1_0.pth',
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--model_weights2', type=str, 
                        default='/research/iprobe-paldebas/Research_Work/Continuous_Learning_MNIST/data/MNIST/Task_2/Task_2_0.pth',
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--model_weights3', type=str, 
                        default='/research/iprobe-paldebas/Research_Work/Continuous_Learning_MNIST/data/MNIST/Task_3/Task_3_0.pth',
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--model_weights4', type=str, 
                        default='/research/iprobe-paldebas/Research_Work/Continuous_Learning_MNIST/data/MNIST/Task_4/Task_4_0.pth',
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--model_weights5', type=str, 
                        default='/research/iprobe-paldebas/Research_Work/Continuous_Learning_MNIST/data/MNIST/Task_5/Task_5_0.pth',
                        help="The path to the file for the model weights (*.pth).")
    
    parser.add_argument('--taskFolder1', type=str, 
                        default='Feature_Extractor/ViT/MNIST/Task_1/',)
    parser.add_argument('--taskFolder2', type=str, 
                        default='Feature_Extractor/ViT/MNIST/Task_2/')
    parser.add_argument('--taskFolder3', type=str, 
                        default='Feature_Extractor/ViT/MNIST/Task_3/')
    parser.add_argument('--taskFolder4', type=str, 
                        default='Feature_Extractor/ViT/MNIST/Task_4/')
    parser.add_argument('--taskFolder5', type=str, 
                        default='Feature_Extractor/ViT/MNIST/Task_5/')
    
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['TORCH_HOME'] = '../TempData/models/'
    main(args)