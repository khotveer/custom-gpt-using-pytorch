import torch

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x = self.data[idx : idx + self.block_size]
        # y = self.data[idx + 1 : idx + 1 + self.block_size]
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + 1 + self.block_size], dtype=torch.long)

        return x, y