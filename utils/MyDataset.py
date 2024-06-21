from torch.utils.data import Dataset


class REDataset(Dataset):
    def __init__(self, data):
        super(REDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
