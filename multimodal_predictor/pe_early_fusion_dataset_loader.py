from torch.utils.data import DataLoader

from pe_early_fusion_dataset import PEEarlyFusionDataset


class PEEarlyFusionDatasetLoader(DataLoader):

    def __init__(
        self,
        dataset_path: str,
        window_size: int = 24,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        print("Loading PE dataset.")
        super(PEEarlyFusionDatasetLoader, self).__init__(
            PEEarlyFusionDataset(
                dataset_path,
                window_size=window_size,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )
