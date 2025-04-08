from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

class OddEvenDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the image and original label from the subset
        image, label = self.subset[idx]
        
        # Convert label to 0 (odd) or 1 (even)
        new_label = 1 if label % 2 == 0 else 0
        return image, new_label

class split_MNIST(torch.utils.data.Dataset):
    def __init__(self, train_test=True):
        super().__init__()
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(root='./data', train=train_test, download=True, transform=transform)
        self.subsets = self.get_split_MNIST_tasks()
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),        
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.subset[index]
        
        # Convert label to 0 (odd) or 1 (even)
        new_label = 1 if label % 2 == 0 else 0
        return image, new_label
    
    def get_split_MNIST_tasks(self):
        task_datasets = []
        for task_idx in range(5):
            # Each task has two consecutive digits (0-1, 2-3, ..., 8-9)
            targets = list(range(2 * task_idx, 2 * task_idx + 2))
            task_indices = [i for i, label in enumerate(self.dataset.targets) if label in targets]
        
            # Subset the dataset for this task
            task_subset = torch.utils.data.Subset(self.dataset, task_indices)
        
            # Wrap subset in custom dataset to relabel as odd/even
            task_datasets.append(task_subset)
    
        return task_datasets
