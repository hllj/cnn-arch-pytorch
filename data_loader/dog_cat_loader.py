from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class DogCatDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.file_length = len(self.file_list)
        return self.file_length

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split("/")[-1].split(".")[0]
        if label == "dog":
            label = 1
        elif label == "cat":
            label = 0
        return img_transformed, label


if __name__ == "__main__":
    import glob
    import os

    train_dir = "data/train"
    test_dir = "data/test"

    train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
    train_list, val_list = train_test_split(train_list, test_size=0.2)
    test_list = glob.glob(os.path.join(test_dir, "*.jpg"))

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_data = DogCatDataset(train_list, transform=train_transforms)
    test_data = DogCatDataset(test_list, transform=test_transforms)
    val_data = DogCatDataset(val_list, transform=val_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=10, shuffle=True)

    print(len(train_data), len(train_loader))
