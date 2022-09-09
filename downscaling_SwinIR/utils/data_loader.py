

# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

from torch.utils.data import DataLoader
from dataset_prep import PrecipDatasetInter
from dataset_temp import TempDatasetInter

def create_loader(file_path: str = None, batch_size: int = 4, patch_size: int = 16,
                 vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "yw_hourly_in"],
                 var_out: list = ["yw_hourly_tar"], sf: int = 10,
                 seed: int = 1234, dataset_type: str = 'precip'):

    """
    file_path : the path to the directory of .nc files
    vars_in   : the list contains the input variable namsaes
    var_out   : the list contains the output variable name
    batch_size: the number of samples per iteration
    patch_size: the patch size for low-resolution image,
                the corresponding high-resolution patch size should be muliply by scale factor (sf)
    sf        : the scaling factor from low-resolution to high-resolution
    seed      : specify a seed so that we can generate the same random index for shuffle function
    dataset_type   : type of the data
    """
    if dataset_type == 'precip':
        dataset = PrecipDatasetInter(file_path, batch_size, patch_size, vars_in, var_out, sf, seed)
    elif dataset_type == 'temperature':
        dataset = TempDatasetInter(file_path, batch_size)
    return DataLoader(dataset, batch_size=None)


