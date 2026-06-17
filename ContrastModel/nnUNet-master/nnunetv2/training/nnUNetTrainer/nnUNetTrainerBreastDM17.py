import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBreastDM17(nnUNetTrainer):
    """nnU-Net trainer with YAML-driven basic hyperparameters for BreastDM17."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = float(os.environ.get("BREASTDM17_NNUNET_LR", self.initial_lr))
        self.weight_decay = float(os.environ.get("BREASTDM17_NNUNET_WEIGHT_DECAY", self.weight_decay))
        self.num_epochs = int(os.environ.get("BREASTDM17_NNUNET_EPOCHS", self.num_epochs))
        self.save_every = int(os.environ.get("BREASTDM17_NNUNET_SAVE_EVERY", self.save_every))

