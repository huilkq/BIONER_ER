from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        self.samples = list(range(1, 101))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = CustomDataset()
    print(len(dataset))
    print(dataset[50])
    print(dataset[1:100])