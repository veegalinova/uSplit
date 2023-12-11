from functools import partial
from pathlib import Path

from usplit.core.data_split_type import DataSplitType
from usplit.data_loader.multifile_raw_dloader import (
    get_train_val_data as get_train_val_data_twochannels,
)


def get_two_channel_files(file_path):
    path = Path(file_path)
    inputs = [Path('synth') / f.name for f in sorted(path.glob('synth/*'))]
    targets = [Path('original') / f.name for f in sorted(path.glob('original/*'))]
    return inputs, targets


def get_train_val_data(
    datadir,
    data_config,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
):
    files_fn = partial(
        get_two_channel_files, file_path=datadir
    )

    return get_train_val_data_twochannels(
        datadir,
        data_config,
        datasplit_type,
        files_fn,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


if __name__ == "__main__":
    from usplit.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict

    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.TwoChannel
    datadir = "/home/vera.galinova/Vera/dexm_train/"
    data = get_train_val_data(
        datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1
    )
    print(len(data))
