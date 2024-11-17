import os
from PIL import Image
from torch.utils.data import Dataset

class FruitData(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.labels = ['Apple', 'avocado', 'Banana', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']
        self.image_paths = []
        self.image_labels = []

        for idx, label in enumerate(sorted(os.listdir(root))):
            label_path = os.path.join(root, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.image_paths.append(img_path)
                    self.image_labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((250, 250), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
        label = self.image_labels[idx]

        return image, label