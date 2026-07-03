import random

from data.collate import default_collate


class DataLoader:

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=None,
    ):
        self.collate_fn = collate_fn or default_collate
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = []
        self.current_index = 0

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        self.indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(self.indices)

        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        start = self.current_index
        end = start + self.batch_size

        batch_indices = self.indices[start:end]
        self.current_index = end

        batch = [
            self.dataset[idx]
            for idx in batch_indices
        ]
        return self.collate_fn(batch)
