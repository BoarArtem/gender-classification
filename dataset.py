from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((224, 224,)), transforms.ToTensor()])

def get_train_dataset(dataset_pth, transform):
    return datasets.ImageFolder(dataset_pth, transform)

def get_test_dataset(dataset_pth, transform):
    return datasets.ImageFolder(dataset_pth, transform)

def get_train_loader(dataset, batch_size = 32, shuffle = True):
    return DataLoader(dataset, batch_size, shuffle)

def get_test_loader(dataset, batch_size = 32, shuffle = False):
    return DataLoader(dataset, batch_size, shuffle)

train_dataset_exc = get_train_dataset("../dataset/train", train_transforms)
train_loader = get_train_loader(train_dataset_exc)

test_dataset_exc = get_test_dataset("../dataset/test", test_transforms)
test_loader = get_test_loader(test_dataset_exc)

# for inputs, labels in train_loader:
#     print(inputs.shape)
#     print(labels.shape)
#
#     break
