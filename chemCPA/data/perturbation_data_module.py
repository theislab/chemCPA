import lightning as L
from torch.utils.data import DataLoader
import torch

def custom_collate(batch):
    transposed = zip(*batch)
    concat_batch = []
    for samples in transposed:
        if samples[0] is None:
            concat_batch.append(None)
        else:
            # Move to CUDA here so that prefetching in the DataLoader yields ready-to-process CUDA tensors
            concat_batch.append(torch.stack(samples, 0).to("cuda"))
    return concat_batch

class PerturbationDataModule(L.LightningDataModule):
    def __init__(self, datasplits, train_bs=32, val_bs=32, test_bs=32):
        super().__init__()
        self.datasplits = datasplits
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs

    def setup(self, stage: str):
        # Assign datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.datasplits["training"]
            self.train_control_dataset = self.datasplits["training_control"]
            self.train_treated_dataset = self.datasplits["training_treated"]
            self.test_dataset = self.datasplits["test"]
            self.test_control_dataset = self.datasplits["test_control"]
            self.test_treated_dataset = self.datasplits["test_treated"]
            self.ood_control_dataset = self.datasplits["test_control"]
            self.ood_treated_dataset = self.datasplits["ood"]

        if stage == "validate" or stage == "test":
            self.test_dataset = self.datasplits["test"]
            self.test_control_dataset = self.datasplits["test_control"]
            self.test_treated_dataset = self.datasplits["test_treated"]

        if stage == "predict":
            self.ood_control_dataset = self.datasplits["test_control"]
            self.ood_treated_dataset = self.datasplits["ood"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bs, shuffle=True, collate_fn=custom_collate)

    def val_dataloader(self):
        return {
            "test": DataLoader(self.test_dataset, batch_size=self.val_bs),
            "test_control": DataLoader(self.test_control_dataset, batch_size=self.val_bs),
            "test_treated": DataLoader(self.test_treated_dataset, batch_size=self.val_bs),
        }

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_bs)

    def predict_dataloader(self):
        return DataLoader(self.ood_dataset, batch_size=self.test_bs)