import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os
import logging
from torchvision import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def get_transforms(dataset_name: str):

    resize_to_224 = True # Variabile per indicare se ridimensionare a 224x224

    if dataset_name in ["cifar10", "cifar100"]:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if resize_to_224:  # Variabile per indicare se ridimensionare a 224x224
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Ridimensiona le immagini
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    elif dataset_name in ["imagenette", "caltech256", "caltech101", "flowers102"]:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if dataset_name in ["caltech256", "caltech101", "flowers102"]:
            train_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))
            test_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))

    else:
        raise ValueError(f"Transforms for dataset {dataset_name} not defined.")
    return train_transform, test_transform

def get_train_and_test_loader(dataset_name: str, data_folder: str = './data', batch_size: int = 64, num_workers: int = 8):

    data_folder = os.path.join(data_folder, dataset_name)
    os.makedirs(data_folder, exist_ok=True)

    dataset_dict = {
        "cifar10": (datasets.CIFAR10, 10),
        "cifar100": (datasets.CIFAR100, 100),
        "imagenette": (datasets.Imagenette, 10),
        "caltech256": (datasets.Caltech256, 257),
        "caltech101": (datasets.Caltech101, 101),
        "flowers102": (datasets.Flowers102, 102),
        #"" : (None,0)
    }

    if dataset_name not in dataset_dict:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are: {list(dataset_dict.keys())}")

    dataset_class, n_cls = dataset_dict[dataset_name]
    train_transform, test_transform = get_transforms(dataset_name)

    # Caricamento e suddivisione dataset
    if dataset_name in ["cifar10", "cifar100"]:
        train_set = dataset_class(root=data_folder, train=True, download=True, transform=train_transform)
        test_set = dataset_class(root=data_folder, train=False, download=True, transform=test_transform)

    elif dataset_name in ["imagenette"]:
        #check if is already downloaded

        download_flag = False   

        if not os.path.exists(data_folder+'/imagenette2'):
            print("Downloading dataset imagenette in path ...",data_folder+'/imagenette2')
            download_flag = True

        train_set = dataset_class(root=data_folder, split='train', download=download_flag, transform=train_transform)
        test_set = dataset_class(root=data_folder, split='val', download=download_flag, transform=test_transform)
        
    elif dataset_name in ["caltech256", "caltech101", "flowers102"]:
        full_dataset = dataset_class(root=data_folder, download=True, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = random_split(full_dataset, [train_size, test_size])
        test_set.dataset.transform = test_transform  # Applica trasformazione di test
    
    elif dataset_name in [""]:
        pass

    # Creazione dei DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Logging
    logger.info(f"{dataset_name} - Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    logger.info(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

    return train_loader, test_loader, n_cls



if __name__ == "__main__":

    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    datasets_list = ["cifar10", "cifar100", "imagenette", "caltech256", "caltech101", "flowers102"]

    for dataset_name in datasets_list:
        dataset_path = './work/project/data'
        batch_size = 32
        num_workers = 8

        trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                            data_folder=dataset_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers)
        
        print(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
        print(f"Number of classes: {n_cls}")
        print("Dataset loaded")
        print("Done")
        
        print(trainloader)
        print(testloader)
        print("Done")
