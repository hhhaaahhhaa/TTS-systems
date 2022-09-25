import random


class EpisodicInfiniteWrapper:
    def __init__(self, dataset, epoch_length):
        self.dataset = dataset
        self.epoch_length = epoch_length

    def __getitem__(self, idx):
        return random.choice(self.dataset)

    def __len__(self):
        return self.epoch_length
