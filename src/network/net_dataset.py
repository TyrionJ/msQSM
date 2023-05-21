from torch import from_numpy
from torch.utils.data import Dataset
import h5py


class NetDataset(Dataset):
    def __init__(self, data_file):
        super(NetDataset, self).__init__()

        self.data_file = data_file

        self.qsm0, self.fld, self.msk = None, None, None
        self.knl = None

        with h5py.File(self.data_file, 'r') as f:
            self.dataset_len = len(f['qsm0'])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.qsm0 is None:
            self.qsm0 = h5py.File(self.data_file, "r")['qsm0']
            self.fld = h5py.File(self.data_file, "r")['fld']
            self.msk = h5py.File(self.data_file, "r")['msk']

        qsm0 = from_numpy(self.qsm0[index]).unsqueeze(dim=0)
        fld = from_numpy(self.fld[index]).unsqueeze(dim=0)
        msk = from_numpy(self.msk[index]).unsqueeze(dim=0)

        return qsm0, fld, msk
