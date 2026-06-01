from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def get_train_dataset(train_dataset_pth, train_transforms):
    return datasets.ImageFolder(root=train_dataset_pth, transform=train_transforms)

def get_test_dataset(test_dataset_pth, test_transforms):
    return datasets.ImageFolder(root=test_dataset_pth, transform=test_transforms)

def get_train_loader(train_dataset, batch_size = 32, shuffle = True):
    return DataLoader(train_dataset, batch_size, shuffle)

def get_test_loader(test_dataset, batch_size = 32, shuffle = False):
    return DataLoader(test_dataset, batch_size, shuffle)

train_dataset_exc = get_train_dataset("../dataset/train", train_transform)
test_dataset_exc = get_test_dataset("../dataset/test", test_transform)

train_dataloader = get_train_loader(train_dataset_exc)


# checking our dataset info
def get_dataset_info(dataloader = train_dataloader):
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)

        break
