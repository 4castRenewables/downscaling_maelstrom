# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2023-01-04"

import os, glob
from typing import List
import socket
import gc
import multiprocessing
from collections import OrderedDict
from timeit import default_timer as timer
import random
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from all_normalizations import ZScore
from tools_utils import CDO
from other_utils import to_list


class HandleDataClass(object):

    def __init__(self, datadir: str, application: str, query: str, purpose: str = None, **kwargs) -> None:
        """
        Initialize Input data object by reading data from netCDF-files
        :param datadir: the directory from where netCDF-files are located (or should be located if downloaded)
        :param application: name of application (must coincide with name in s3-bucket)
        :param query: query string which can be used to load data from the s3-bucket of the application
        :param purpose: optional name to indicate the purpose of queried data (used as key for the data-dictionary)
        """
        method = HandleDataClass.__init__.__name__

        self.host = os.getenv("HOSTNAME") if os.getenv("HOSTNAME") is not None else "unknown"
        purpose = query if purpose is None else purpose
        self.application = application
        self.datadir = datadir
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
        self.ldownload_last = None

        self.data, self.timing, self.data_info = self.handle_data_req(query, purpose, **kwargs)

    def handle_data_req(self, query: str, purpose, **kwargs):
        """
        Handles a data-query by parsing it to the get_data function
        :param query: the query-string to submit to the climetlab-API of the application
        :param purpose: the name/purpose of the retireved data (used to append the data-dictionary)
        :return: the xr.Dataset retireved from get_data and dictionaries for the loading time and the memory consumption
        """
        method = HandleDataClass.handle_data_req.__name__

        datafile = os.path.join(self.datadir, "{0}_{1}.nc".format(self.application, purpose))
        self.ldownload_last = self.set_download_flag(datafile)
        # time data retrieval
        t0_load = timer()
        ds = self.get_data(query, datafile, **kwargs)
        load_time = timer() - t0_load
        if self.ldownload_last:
            print("%{0}: Downloading took {1:.2f}s.".format(method, load_time))
            _ = HandleDataClass.ds_to_netcdf(ds, datafile)

        data = OrderedDict({purpose: ds})
        timing = {"loading_times": {purpose: load_time}}
        data_info = {"memory_datasets": {purpose: ds.nbytes}}

        return data, timing, data_info

    def append_data(self, query: str, purpose: str = None, **kwargs):
        """
        Appends data-dictionary of the class and also tracks basic benchmark parameters
        :param query: the query-string to submit to the climetlab-API of the application
        :param purpose: the name/purpose of the retireved data (used to append the data-dictionary)
        :return: appended self.data-dictionary with {purpose: xr.Dataset}
        """
        purpose = query if purpose is None else purpose
        ds_app, timing_app, data_info_app = self.handle_data_req(query, purpose, **kwargs)

        self.data.update(ds_app)
        self.timing["loading_times"].update(timing_app["loading_times"])
        self.data_info["memory_datasets"].update(data_info_app["memory_datasets"])
        
    def set_download_flag(self, datafile):
        """
        Depending on the hosting system and on the availability of the dataset on the filesystem
        (stored under self.datadir), the download flag is set to False or True. Also returns a dictionary for the
        respective netCDF-filenames.
        :return: Boolean flag for downloading and dictionary of data-filenames
        """
        method = HandleDataClass.set_download_flag.__name__

        ldownload = True if "login" in self.host else False        
        stat_file = os.path.isfile(datafile)

        if stat_file and ldownload:
            print("%{0}: Datafiles are already available under '{1}'".format(method, self.datadir))
            ldownload = False
        elif not stat_file and not ldownload:
            raise ValueError("%{0}: Data is not available under '{1}',".format(method, self.datadir) +
                             "but downloading on computing node '{0}' is not possible.".format(self.host))

        return ldownload

    def get_data(self, *args):
        """
        Function to either downlaod data from the s3-bucket or to read from file.
        """
        raise NotImplementedError("Please set-up a customized get_data-function.")

    @staticmethod
    def reshape_ds(ds):
        """
        Convert a xarray dataset to a data-array where the variables will constitute the last dimension (channel last)
        :param ds: the xarray dataset with dimensions (dims)
        :return da: the data-array with dimensions (dims, variables)
        """
        da = ds.to_array(dim="variables")
        da = da.transpose(..., "variables")
        return da

    @staticmethod
    def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
        """
        Split data array with variables-dimension into input and target data for downscaling.
        :param da: The unsplitted data array.
        :param target_var: Name of target variable which should consttute the first channel
        :return: The splitted data array.
        """
        invars = [var for var in da["variables"].values if var.endswith("_in")]
        tarvars = [var for var in da["variables"].values if var.endswith("_tar")]

        # ensure that ds_tar has a channel coordinate even in case of single target variable
        roll = False
        if len(tarvars) == 1:
            sl_tarvars = tarvars
        else:
            sl_tarvars = slice(*tarvars)
            if tarvars[0] != target_var:     # ensure that target variable appears as first channel
                roll = True

        da_in, da_tar = da.sel({"variables": invars}), da.sel(variables=sl_tarvars)
        if roll: da_tar = da_tar.roll(variables=1, roll_coords=True)

        return da_in, da_tar

    @staticmethod
    def gather_monthly_netcdf(file_list: List, nfiles_resampled: int = 36, loverwrite: bool = True):
        """
        Merges monthyl netCDF-files to larger netCDF-files to optiize building TensorFlow datasets later on.
        Files are stored in a tmp-subdirectory of self.datadir
        :param file_list: List of files to gather/merge
        :param nfiles_resampled: number of files to merge to single netCDF files
                                 (nfiles_resampled must be a divisor of len(file_list)).
        :param loverwrite: Boolean to overwrite temp-data
        """
        cdo = CDO()

        file_list = to_list(file_list)
        nfiles_in = len(file_list)
        nfiles_out = int(nfiles_in/nfiles_resampled)
        if np.mod(nfiles_in, nfiles_resampled) != 0:
            raise ValueError(f"nfiles_resampled ({nfiles_resampled:d}) must be a divisor of the total number" +
                             f" of files ({nfiles_in:d})")

        file_list_loc = random.shuffle(file_list)

        datadir = os.path.dirname(file_list[0])
        tmp_dir = os.path.join(datadir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        for i in range(nfiles_out):
            files2merge = file_list_loc[i*nfiles_resampled:(i+1)*nfiles_resampled]

            fname_now = os.path.join(tmp_dir, f"ds_resampled_{i:0d}.nc")
            if os.path.isfile(fname_now):
                mess = f"netCDF-file '{fname_now}' already exists."
                if loverwrite:
                    print(f"{mess} Existing file is overwritten.")
                else:
                    raise FileExistsError(mess)
            print(f"Write merged data to file '{fname_now}'.")

            cdo.run(files2merge + [fname_now], OrderedDict([("mergetime", "")]))

    def make_shuffled_dataset(self, ds: xr.Dataset, batch_size: int, sample_dim: str = "time",  nfiles: int = None,
                              samples_per_file: int = None, loverwrite: bool = True):

        datadir_shuffled = os.path.join(self.datadir, "tmp")
        os.makedirs(datadir_shuffled, exist_ok=True)

        sample_dim_data = ds[sample_dim].copy(deep=True)
        nsampels = np.shape(sample_dim_data)[0]
        ds = ds.rename_dims({sample_dim: "sample_ind"})
        ds["sample_ind"] = range(np.shape(nsampels)[0])

        inds = np.arange(nsampels)
        np.random.shuffle(inds)

        if nfiles is not None:
            samples_per_file = int(nsampels/12)
        else:
            if samples_per_file is None:
                raise ValueError("Either provide nfiles or samples_per_file-argument.")

        if np.mod(samples_per_file, batch_size) > 0:
            n = int(samples_per_file/batch_size)
            samples_per_file = n*batch_size
            print(f"Batch size is not a divider of samples per file. " +
                  f"samples_per_file is adjusted to {samples_per_file:d}")

        dropped_samples = np.mod(nsampels, samples_per_file)

        if dropped_samples > 0:
            print(f"{dropped_samples} samples will be dropped.")

        for i in range(int(nsampels / samples_per_file)):
            inds_now = inds[i * samples_per_file: (i + 1) * samples_per_file]
            print(f"Load data to memory for {i + 1:d}th subset...")
            ds_subset = ds.isel({"sample_ind": inds_now}).load()
            print("Data loaded sucsessfully!")

            nsamples_now = np.shape(ds_subset["sample_ind"])[0]
            ds_subset["sample_ind"] = range(i * nsamples_now, (i + 1) * nsamples_now)
            fname_now = os.path.join(datadir_shuffled, f"ds_resampled_{i:0d}.nc")

            if os.path.isfile(fname_now):
                mess = f"netCDF-file '{fname_now}' already exists."
                if loverwrite:
                    print(f"{mess} Existing file is overwritten.")
                else:
                    raise FileExistsError(mess)
            print(f"Write data subset to file '{fname_now}'.")
            ds_subset.to_netcdf(fname_now)

    @staticmethod
    def make_tf_dataset_dyn(datadir: str, file_patt: str, batch_size: int, norm_dims: List, lshuffle: bool = True,
                            nshuffle_per_file: int = None, lprefetch: bool = True, nworkers: int = None,
                            selected_predictors: List = ()):
        """
        Build TensorFlow dataset from netCDF-files processed by make_shuffled_dataset or gather_monthly_netcdf.
        In contrast to make_tf_dataset_allmem, this data streaming approach is memory-efficient, meaning that the
        complete dataset is NOT loaded into memory.
        :param datadir: directory where netCDF-files for TF dataset are strored
        :param file_patt: filename pattern to glob files from datadir
        :param norm_dims: names of dimension over which normalization is applied
        :param batch_size: desired mini-batch size
        :param lshuffle: boolean to enable sample shuffling (to be used when data was prepared
                                                             with gather_monthly_netcdf)
        :param nshuffle_per_file: buffer for shuffling operation
        :param lprefetch: boolean to enable prefetching
        :param nworkers: number of workers to stream from netCDF-files
        :param selected_predictors: List of selected predictor variables
        :return: TensorFlow dataset object
        """

        if nworkers is None:
            nworkers = multiprocessing.cpu_count()

        ds_obj = StreamMonthlyNetCDF(os.path.join(datadir, "tmp"), file_patt, norm_dims, workers=int(nworkers),
                                     var_tar2in="hsurf_tar", selected_predictors=selected_predictors)

        tf_fun1 = lambda fname: tf.py_function(ds_obj.read_netcdf, [fname], tf.bool)
        tf_fun2 = lambda i: tf.numpy_function(ds_obj.getitems, [i], (tf.float32, tf.float32))

        tfds = tf.data.Dataset.from_tensor_slices(ds_obj.file_list).map(tf_fun1)
        tfds_range = tf.data.Dataset.range(ds_obj.samples_per_file)
        if lshuffle:
            nshuffle_per_file = ds_obj.samples_per_file if nshuffle_per_file is None else nshuffle_per_file
            tfds_range = tfds.shuffle(nshuffle_per_file)
        tfds = tfds.interleave(lambda x: tfds_range.batch(nworkers).map(tf_fun2).unbatch().batch(batch_size))

        if lprefetch:
            tfds = tfds.prefetch(int(ds_obj.samples_per_file/2))

        return tfds.repeat()

    @staticmethod
    def make_tf_dataset_allmem(da: xr.DataArray, batch_size: int, lshuffle: bool = True, shuffle_samples: int = 20000,
            named_targets: bool = False, var_tar2in: str = None, lembed: bool = False) -> tf.data.Dataset:
        """
        Build-up TensorFlow dataset from a generator based on the xarray-data array.
        NOTE: All data is loaded into memory.
        :param da: the data-array from which the dataset should be cretaed. Must have dimensions [time, ..., variables].
                   Input variable names must carry the suffix '_in', whereas it must be '_tar' for target variables.
        :param batch_size: number of samples per mini-batch
        :param lshuffle: flag if shuffling should be applied to dataset
        :param shuffle_samples: number of samples to load before applying shuffling
        :param named_targets: flag if target of TF dataset should be dictionary with named target variables
        :param var_tar2in: name of target variable to be added to input (used e.g. for adding high-resolved topography to the input)
        :param lembed: flag to trigger temporal embedding (not implemented yet!)
        """
        da = da.load()
        da_in, da_tar = HandleDataClass.split_in_tar(da)
        if var_tar2in is not None:
            da_in = xr.concat([da_in, da_tar.sel({"variables": var_tar2in})], "variables")

        varnames_tar = da_tar["variables"].values

        def gen_named(darr_in, darr_tar):
            # darr_in, darr_tar = darr_in.load(), darr_tar.load()
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                tar_now = darr_tar.isel({"time": t})
                yield tuple((darr_in.isel({"time": t}).values,
                             {var: tar_now.sel({"variables": var}).values for var in varnames_tar}))

        def gen_unnamed(darr_in, darr_tar):
            # darr_in, darr_tar = darr_in.load(), darr_tar.load()
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                yield tuple((darr_in.isel({"time": t}).values, darr_tar.isel({"time": t}).values))

        if named_targets is True:
            gen_now = gen_named
        else:
            gen_now = gen_unnamed

        # create output signatures from first sample
        s0 = next(iter(gen_now(da_in, da_tar)))
        sample_spec_in = tf.TensorSpec(s0[0].shape, dtype=s0[0].dtype)
        if named_targets is True:
            sample_spec_tar = {var: tf.TensorSpec(s0[1][var].shape, dtype=s0[1][var].dtype) for var in varnames_tar}
        else:
            sample_spec_tar = tf.TensorSpec(s0[1].shape, dtype=s0[1].dtype)

        # re-instantiate the generator and build TF dataset
        gen_train = gen_now(da_in, da_tar)

        if lembed is True:
            raise ValueError("Time embedding is not supported yet.")
        else:
            data_iter = tf.data.Dataset.from_generator(lambda: gen_train,
                                                       output_signature=(sample_spec_in, sample_spec_tar))

        # Notes:
        # * cache is reuqired to make repeat work properly on datasets based on generators
        #   (see https://stackoverflow.com/questions/60226022/tf-data-generator-keras-repeat-does-not-work-why)
        # * repeat must be applied after shuffle to get varying mini-batches per epoch
        # * batch-size is increaded to allow substepping in train_step
        if lshuffle is True:
            data_iter = data_iter.cache().shuffle(shuffle_samples).batch(batch_size, drop_remainder=True).repeat()
        else:
            data_iter = data_iter.cache().batch(batch_size, drop_remainder=True).repeat()

        # clean-up to free some memory
        del da
        gc.collect()

        return data_iter

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, fname: str, comp_lvl=5):
        """
        Create dictionary for compressing all variables of dataset in netCDF-files
        :param ds: the xarray-dataset
        :param fname: name of the target netCDF-file
        :param comp_lvl: the compression level
        :return: True in case of success
        """
        method = HandleDataClass.ds_to_netcdf.__name__

        comp = dict(zlib=True, complevel=comp_lvl)
        try:
            encoding_ds = {var: comp for var in ds.data_vars}
            print("%{0}: Save dataset to netCDF-file '{1}'".format(method, fname))
            ds.to_netcdf(path=fname, encoding=encoding_ds)  # , engine="scipy")
        except Exception as err:
            print("%{0}: Failed to handle and save input dataset.".format(method))
            raise err

        return True

    @staticmethod
    def has_internet():
        """
        Checks if Internet connection is available.
        :return: True if connected, False else.
        """
        try:
            # connect to the host -- tells us if the host is actually
            # reachable
            socket.create_connection(("1.1.1.1", 53), timeout=5)
            return True
        except OSError:
            pass
        return False


def get_dataset_filename(datadir: str, dataset_name: str, subset: str, laugmented: bool = False):

    allowed_subsets = ("train", "val", "test")

    if subset in allowed_subsets:
        pass
    else:
        raise ValueError(f"Unknown dataset subset '{subset}' chosen. Allowed subsets are {*allowed_subsets,}")

    fname_suffix = "downscaling"

    if dataset_name == "tier1":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if laugmented: fname_suffix = f"{fname_suffix}_aug"
    elif dataset_name == "tier2":
        fname_suffix = f"{fname_suffix}_{dataset_name}_{subset}"
        if laugmented: raise ValueError("No augmented dataset available for Tier-2.")
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}' passed.")

    ds_filename = os.path.join(datadir, f"{fname_suffix}.nc")

    if not os.path.isfile(ds_filename):
        raise FileNotFoundError(f"Could not find requested dataset file '{ds_filename}'")

    return ds_filename


class StreamMonthlyNetCDF(object):
    # TO-DO:
    # - check if normalization works from dataset rather than data array
    # - save normalization parameters to file

    def __init__(self, datadir, patt, norm_dims: List, workers=4, sample_dim: str = "time",
                 selected_predictors: List = None, var_tar2in: str = None, samples_per_file: int = 8640):
        self.data_dir = datadir
        self.file_list = patt
        self.ds = xr.open_mfdataset(list(self.file_list))  # , parallel=True)
        self.data_norm = ZScore(norm_dims)
        self.norm_params = self.data_norm.get_required_stats(self.ds)
        self.sample_dim = sample_dim
        self.times = self.ds[sample_dim].load()
        self.nsamples = self.ds.dims[sample_dim]
        self.variables = list(self.ds.variables)
        self.samples_per_file = samples_per_file
        self.predictor_list = selected_predictors
        self.var_tar2in = var_tar2in
        self.predictors_now, self.predictands_now = None, None

        print(f"Number of used workers: {workers:d}")
        self.pool = multiprocessing.pool.ThreadPool(workers)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, i):
        data = self.index_to_sample(i)
        return data

    def getitems(self, indices):
        return np.array(self.pool.map(self.__getitem__, indices))

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, datadir):
        if not os.path.isdir(datadir):
            raise NotADirectoryError(f"Parsed data directory '{datadir}' does not exist.")

        self._data_dir = datadir

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, patt):
        patt = patt if patt.endswith(".nc") else f"{patt}.nc"
        files = glob.glob(os.path.join(self.data_dir, patt))

        if not files:
            raise FileNotFoundError(f"Could not find any files with pattern '{patt}' under '{self.data_dir}'.")

        self._file_list = np.asarray(sorted(files, key=lambda s: int(re.search(r'\d+', os.path.basename(s)).group())))

    @property
    def sample_dim(self):
        return self._sample_dim

    @sample_dim.setter
    def sample_dim(self, sample_dim):
        if not sample_dim in self.ds.dims:
            raise KeyError(f"Could not find dimension '{sample_dim}' in data.")

        self._sample_dim = sample_dim

    @property
    def predictor_list(self):
        return self._predictor_list

    @predictor_list.setter
    def predictor_list(self, selected_predictors):
        stat_list = [predictor in self.variables for predictor in selected_predictors]
        if not all(stat_list):
            miss_inds = [i for i, x in enumerate(stat_list) if x]
            miss_vars = [selected_predictors[i] for i in miss_inds]
            raise ValueError(f"Could not find the following predictor variables in dataset: {*miss_vars,}")

        self._predictor_list = selected_predictors

    def read_netcdf(self, fname):
        fname = tf.keras.backend.get_value(fname)
        fname = str(fname).lstrip("b'").rstrip("'")
        print(f"Load data from {fname}...")
        ds_now = xr.open_dataset(str(fname), engine="netcdf4")
        ds_now = self.data_norm(ds_now)
        da_now = HandleDataClass.reshape_ds(ds_now.astype("float32", copy=False))
        predictors_now, self.predictands_now = HandleDataClass.split_in_tar(da_now)
        if self.predictor_list is not None:
            self.predictors_now = predictors_now.sel({"variables": self.predictor_list})
        else:
            self.predictors_now = predictors_now

        return True

    def index_to_sample(self, index):

        in_sample = self.predictors_now.isel({self.sample_dim: index}).astype("float32", copy=False)
        tar_sample = self.predictands_now.isel({self.sample_dim: index}).astype("float32", copy=False)

        if self.var_tar2in is not None:
            in_sample = xr.concat([in_sample, tar_sample.sel({"variables": self.var_tar2in})], "variables")

        return in_sample.to_array(), tar_sample.to_array()






