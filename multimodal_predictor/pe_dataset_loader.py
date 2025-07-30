from torch.utils.data import DataLoader

from pe_dataset import PEDataset


class PEDatasetLoader(DataLoader):

    def __init__(
        self,
        dataset_path: str,
        ehr_scaler_path: str,
        ehr_features: list,
        window_size: int = 24,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        print("Loading PE dataset.")
        super(PEDatasetLoader, self).__init__(
            PEDataset(
                dataset_path, ehr_scaler_path, ehr_features, window_size=window_size
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )
