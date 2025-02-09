# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory
import random
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, ShuffleSplit, StratifiedShuffleSplit
from mne.io.edf import read_raw_edf, read_raw_gdf
from mne.io import concatenate_raws, read_raw_cnt, read_raw_nirx
from mne.channels import make_standard_montage, read_custom_montage
from mne import Epochs, events_from_annotations, pick_types

import torch
from torch import nn
import datetime
import time
import math
import os
import gc

use_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if use_CUDA else 'cpu')

dataset_name = "FineMI"
class_of_interest = ["1", "6"]
data_type = ["EEG"]

preprocessing_params = {
    "use_CAR": True,
    "use_band_pass": True,
    "moving_average_std": True,
    "n_jobs": 2
    # "n_jobs": 4
}

band_pass_params = {
    "iir_filter": True,
    "filter_order": 3,
    "filter_type": "butter",
    "lower_bound_eeg": 4,
    "upper_bound_eeg": 40.
}

for item in band_pass_params:
    preprocessing_params[item] = band_pass_params[item]

data_augment_params = {
    "use_data_augment": True,
    "augment_times": 6,
    "method": "white_noise_data_augment",
}
noise_adding_params = {
    "white_noise_std": 0.02,
    "white_noise_snr": 1.,
}

for item in noise_adding_params:
    data_augment_params[item] = noise_adding_params[item]

use_models = {
    "ShallowConvNet": True,
}

params = {
    "ShallowConvNet": {
        "F1": 40,
        "D": 1,
        "conv1_size": 25,
        "avg_pool_size": 75,
        "avg_pool_stride": 15,
        "max_norm_temporal_conv": 2.0,
        "max_norm_spatial_conv": 2.0,
        "max_norm_linear": 0.5,
        "learning_rate": 1e-3,
        "dropout_rate": 0.5,
        "validation_stopping_patience": 200,
    },
}

training_params = {
    "dataset_name": dataset_name,
    "use_CUDA": use_CUDA,
    "device": device,
    "save_best": True,
    "n_epochs": 800,
    "optimizer_type": "Adam",
    "batch_size": 64,
    "early_stop": True,
    "early_stop_monitor": "val_loss",
    "checkpoint_monitor": "val_loss",
}
params["DL_Training"] = training_params

one_time = True

scenario_name = "intra_subject_cross_validation"

K = 5
valid_size = 0.2

plot_params = {}

seed = 1
n_jobs = 4
mne.set_log_level(verbose="WARNING")
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Remove randomness (maybe slower on Tesla GPUs)
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 1:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FineMI():
    def __init__(self, tmin=0, tmax=4, baseline_tmin=-2, baseline_tmax=0, class_of_interest=None, down_sample=False,
                 down_sample_rate=250, resample_fnirs=False, resample_rate_fnirs=250):

        self.name = "FineMI"

        if class_of_interest is None:
            class_of_interest = ["1", "5"]

        self.subject_list = []
        self.use_all_subject_sessions = True
        self.subject_session_names_included = []

        self.exclude_trials = []

        self.raw_file_train_list = []
        self.raw_file_fnirs_list = []

        self.data_train = []
        self.label_train = []

        self.tmin = tmin
        self.tmax = tmax

        self.baseline_tmin = baseline_tmin
        self.baseline_tmax = baseline_tmax

        self.down_sample = down_sample
        self.down_sample_rate = down_sample_rate
        self.resample_fnirs = resample_fnirs
        self.resample_rate_fnirs = resample_rate_fnirs

        self.sample_rate = 1000
        self.sample_rate_fnirs = 7.8125
        self.n_subjects = 18
        self.n_sessions_per_subject = 1
        self.n_electrodes = 62
        self.channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                              'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                              'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                              'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
                              'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
        self.n_classes = 8
        self.n_blocks = 8
        self.tasks = [
            {
                "joint": "Hand",
                "move": "flexion_extension"
            },
            {
                "joint": "Wrist",
                "move": "flexion_extension"
            },
            {
                "joint": "Wrist",
                "move": "adduction_abduction"
            },
            {
                "joint": "Elbow",
                "move": "pronation_supination"
            },
            {
                "joint": "Elbow",
                "move": "flexion_extension"
            },
            {
                "joint": "Shoulder",
                "move": "pronation_supination"
            },
            {
                "joint": "Shoulder",
                "move": "adduction_abduction"
            },
            {
                "joint": "Shoulder",
                "move": "flexion_extension"
            }
        ]

        class_names = [task["joint"] + "_" + task["move"] for task in self.tasks]
        if class_of_interest is not None:
            class_names_of_interest = []
            for class_idx, class_name in enumerate(class_names):
                if str(class_idx + 1) in class_of_interest:
                    class_names_of_interest.append(class_name)
            self.class_names = class_names_of_interest
        else:
            self.class_names = class_names

        self.path = ""

        self.event_id_train = {}
        self.event_id_fnirs = {}
        event_id_train = {
            "1": 1,  # hand open/close
            "2": 2,  # wrist flexion/extension
            "3": 3,  # wrist abduction/adduction
            "4": 4,  # elbow pronation/supination
            "5": 5,  # elbow flexion/extension
            "6": 6,  # shoulder pronation/supination
            "7": 7,  # shoulder abduction/adduction
            "8": 8,  # shoulder flexion/extension
        }
        event_id_fnirs = {
            "1.0": 1,  # hand open/close
            "2.0": 2,  # wrist flexion/extension
            "3.0": 3,  # wrist abduction/adduction
            "4.0": 4,  # elbow pronation/supination
            "5.0": 5,  # elbow flexion/extension
            "6.0": 6,  # shoulder pronation/supination
            "7.0": 7,  # shoulder abduction/adduction
            "8.0": 8,  # shoulder flexion/extension
        }
        if class_of_interest is not None:
            for c in class_of_interest:
                self.event_id_train[c] = event_id_train[c]
                c = c + ".0"
                self.event_id_fnirs[c] = event_id_fnirs[c]
        else:
            self.event_id_train = event_id_train
            self.event_id_fnirs = event_id_fnirs

    def load(self, subject_list=[1], session_list=[1], path="./Data/FineMI/", data_type=None):

        assert len(subject_list) > 0, "Use at least one subject!"

        self.raw_file_train_list = []
        self.raw_file_fnirs_list = []

        self.subject_list = subject_list
        self.path = path

        channels_to_drop = ["M1", "M2", "HEO", "VEO", "EKG", "EMG"]

        for subject in subject_list:

            if "EEG" in data_type:
                subject_session_name = str(subject)
                raw_file_list = []

                if subject_session_name == "1":  # in subject 1, the
                    # subject 1
                    # Block1-4
                    file_name = "../Data/FineMI/subject" + subject_session_name + "/EEG/block1-4.cnt"
                    raw_file = read_raw_cnt(file_name, preload=True)
                    idx_to_remove = np.arange(-40, 0)  # Delete the last 40 trials
                    raw_file.annotations.delete(idx_to_remove)
                    raw_file.crop_by_annotations()
                    raw_file_list.append(raw_file)

                    # Block5-8
                    for block_idx in range(4, self.n_blocks):
                        file_name = "../Data/FineMI/subject" + subject_session_name + "/EEG/block" + str(
                            block_idx + 1) + ".cnt"
                        raw_file = read_raw_cnt(file_name, preload=True)
                        raw_file_list.append(raw_file)
                else:
                    # other subjects
                    for block_idx in range(self.n_blocks):
                        file_name = "../Data/FineMI/subject" + subject_session_name + "/EEG/block" + str(
                            block_idx + 1) + ".cnt"
                        raw_file = read_raw_cnt(file_name, preload=True)
                        if subject_session_name == "5" and block_idx == 5:  # delete the first trial in Block6 of subject 5
                            raw_file.annotations.delete(0)
                            raw_file.crop_by_annotations()
                        raw_file_list.append(raw_file)

                raw_file_train = concatenate_raws(raw_file_list)
                # load the location of electrodes and add into the Raw object
                self.montage = read_custom_montage("../Data/FineMI/channel_location_64_neuroscan.locs")
                raw_file_train.set_montage(self.montage, on_missing="warn")

                raw_file_train.drop_channels(channels_to_drop)
                self.raw_file_train_list.append(raw_file_train)

            if "fNIRS" in data_type:
                raw_file_list = []

                subject_session_name = str(subject)
                print("Subject session: subject", subject_session_name)
                if subject_session_name == "1":
                    # subject 1
                    # Block1-4
                    file_name = "../Data/FineMI/subject" + subject_session_name + "/fNIRS/block1-4"
                    raw_file = read_raw_nirx(file_name, preload=True)

                    idx_to_remove = np.arange(-40, 0)  # Delete the last 40 trials
                    raw_file.annotations.delete(idx_to_remove)
                    raw_file.crop_by_annotations()
                    raw_file_list.append(raw_file)

                    # Block5-8
                    for block_idx in range(4, self.n_blocks):
                        file_name = "../Data/FineMI/subject" + subject_session_name + "/fNIRS/block" + str(
                            block_idx + 1)
                        raw_file = read_raw_nirx(file_name, preload=True)
                        raw_file_list.append(raw_file)
                else:
                    # other sessions
                    for block_idx in range(self.n_blocks):
                        file_name = "../Data/FineMI/subject" + subject_session_name + "/fNIRS/block" + str(
                            block_idx + 1)
                        raw_file = read_raw_nirx(file_name, preload=True)
                        if subject_session_name == "5" and block_idx == 5:  # delete the first trial in Block6 of subject 5
                            raw_file.annotations.delete(0)
                            raw_file.crop_by_annotations()
                        raw_file_list.append(raw_file)

                raw_file_fnirs = concatenate_raws(raw_file_list)
                self.raw_file_fnirs_list.append(raw_file_fnirs)


dataset = FineMI(tmin=0, tmax=4, baseline_tmin=-4, baseline_tmax=-2, class_of_interest=class_of_interest,
                 down_sample=True, down_sample_rate=250)

start_subject = 1
n_subjects = 18
end_subject = n_subjects
target_subject = 3


def common_average_reference(inst):
    """
    :param inst:
    :return:
    """
    raw_file, _ = mne.set_eeg_reference(inst, ref_channels='average')
    return raw_file


def band_pass_filter(raw_file, lower_bound, upper_bound, filter_type="butter", filter_order=3, picks=None):
    iir_params = {
        "output": "sos",
        "order": filter_order,
        "ftype": filter_type
    }
    raw_file.filter(lower_bound, upper_bound, method='iir', picks=picks, iir_params=iir_params)
    return raw_file


def exponential_moving_standardize(data, factor_new=0.001, eps=1e-4):
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    return standardized.T


def extract_epoch(raw_file, events, event_id, tmin, tmax, sample_rate=1000, down_sample=False,
                  down_sample_rate=1000, n_jobs=1, picks=None, baseline_tmin=None, baseline_tmax=None):
    if baseline_tmin is None and baseline_tmax is None:
        baseline = None
    else:
        baseline = (baseline_tmin, baseline_tmax)
    epochs = Epochs(raw_file, events, event_id, tmin, tmax, proj=True, picks=picks,
                    # baseline=None, preload=True, verbose=False)
                    baseline=baseline, preload=True, verbose=False)

    if down_sample and down_sample_rate != sample_rate:
        epochs.resample(down_sample_rate, npad='auto', n_jobs=n_jobs)

    return epochs


def get_epochs_data(raw_file, dataset, event_id, subject=1, load_test_label=False, picks=None, session_idx=0,
                    data_type="EEG"):
    events, _ = events_from_annotations(raw_file, event_id=event_id)
    tmin = dataset.tmin
    tmax = dataset.tmax
    baseline_tmin = None
    baseline_tmax = None

    epochs = extract_epoch(raw_file, events, event_id, tmin, tmax, dataset.sample_rate,
                           dataset.down_sample, dataset.down_sample_rate, picks=picks,
                           baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax)
    data = epochs.get_data()
    label = epochs.events[:, -1]

    label_names = np.unique(label)
    for label_idx, label_name in enumerate(label_names):
        label[label == label_name] = label_idx

    n_samples = data.shape[-1]
    if data_type == "EEG":
        if dataset.down_sample:
            if (n_samples - (dataset.tmax - dataset.tmin) * dataset.down_sample_rate) > 0:
                remainder = int(n_samples - (dataset.tmax - dataset.tmin) * dataset.down_sample_rate)
                if remainder != 0:
                    data = data[:, :, :-remainder]
        elif (n_samples - (dataset.tmax - dataset.tmin) * dataset.sample_rate) > 0:
            remainder = int(n_samples - (dataset.tmax - dataset.tmin) * dataset.sample_rate)
            if remainder != 0:
                data = data[:, :, :-remainder]

    return data, label, epochs.info


def preprocessing_func(raw_file, dataset, model_name, preprocessing_params, load_test_label=False, subject_idx=0,
                       session_idx=0, seed=1):
    if preprocessing_params["use_CAR"]:
        raw_file = common_average_reference(raw_file)

    if preprocessing_params["use_band_pass"]:
        frequency_bands_list = [
            {"fmin": preprocessing_params["lower_bound_eeg"], "fmax": preprocessing_params["upper_bound_eeg"]}
        ]
    else:
        frequency_bands_list = [
            {"fmin": raw_file.info["highpass"], "fmax": raw_file.info["lowpass"]}
        ]
    print("Frequency list: ", str(frequency_bands_list))
    n_filters = len(frequency_bands_list)
    X_total = []
    Y = []

    for i in range(n_filters):
        f_min = frequency_bands_list[i]["fmin"]
        f_max = frequency_bands_list[i]["fmax"]

        filter_type = preprocessing_params["filter_type"]
        if preprocessing_params["use_band_pass"]:
            filtered_raw_file = band_pass_filter(raw_file.copy(), f_min, f_max,
                                                 filter_type=filter_type,
                                                 filter_order=preprocessing_params["filter_order"])
        else:
            filtered_raw_file = raw_file.copy()

        if preprocessing_params["moving_average_std"]:
            filtered_raw_file = filtered_raw_file.apply_function(
                exponential_moving_standardize,
                picks="eeg",
                n_jobs=preprocessing_params["n_jobs"],
                channel_wise=False)

        event_id = dataset.event_id_test if load_test_label else dataset.event_id_train
        picks = pick_types(filtered_raw_file.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        X, Y, info = get_epochs_data(filtered_raw_file, dataset,
                                     event_id, subject=subject_idx + 1, session_idx=session_idx, picks=picks,
                                     load_test_label=load_test_label)

        if i == 0:
            X_total = np.expand_dims(X, axis=1)

        else:
            X_total = np.concatenate([X_total, np.expand_dims(X, axis=1)], axis=1)

    X = np.squeeze(X_total)
    return X, Y, info, frequency_bands_list


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels, conv1_size=25, avg_pool_size=75, avg_pool_stride=15, F1=40, D=1,
                 max_norm_temporal_conv=2.0, max_norm_spatial_conv=2.0, max_norm_linear=0.5,
                 in_features_length=27, dropout_rate=0.5, n_classes=4):
        super(ShallowConvNet, self).__init__()
        self.conv1_size = conv1_size
        self.avg_pool_size = avg_pool_size
        self.avg_pool_stride = avg_pool_stride
        assert (n_channels is not None)
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.temporal_conv = Conv2dWithConstraint(in_channels=1, out_channels=F1, kernel_size=(1, self.conv1_size),
                                                  max_norm=max_norm_temporal_conv)
        self.spatial_conv = Conv2dWithConstraint(in_channels=F1, out_channels=F1, kernel_size=(self.n_channels, 1),
                                                 bias=False, max_norm=max_norm_spatial_conv)
        self.bn = nn.BatchNorm2d(F1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, self.avg_pool_size), stride=(1, self.avg_pool_stride))
        self.drop = nn.Dropout(p=self.dropout_rate)

        self.fc = LinearWithConstraint(in_features=F1 * in_features_length, out_features=n_classes,
                                       max_norm=max_norm_linear)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        n_trials, n_channels, n_times = x.shape
        y = self.temporal_conv(x.view(n_trials, 1, n_channels, n_times))
        y = self.spatial_conv(y)
        y = self.bn(y)
        y = torch.square(y)
        y = self.avg_pool(y)
        y = torch.log(torch.clip(y, min=1e-6))
        y = self.drop(y)
        y = self.fc(y.view(y.shape[0], -1))
        y = self.softmax(y)
        return y


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' create success')
        return True
    else:
        print(path + ' already exist')
        return False


def get_white_noise(X, white_noise_std, snr=None, eps=1e-7):
    if snr is not None:
        snr = 10 ** (snr / 10.0)
        xpower = np.mean(X ** 2, axis=-1, keepdims=True)
        npower = xpower / snr
    else:
        npower = 1
    return np.random.normal(0, white_noise_std, size=X.shape) * np.sqrt(npower)


def data_augment(X, Y, subject_session_name, dataset_name, data_augment_params, X_idx=None, seed=1, info=None):
    n_classes = len(np.unique(Y))
    new_X_list = []
    new_Y_list = []

    if data_augment_params["method"] == "white_noise_data_augment":
        print("Augment Time: ", data_augment_params["augment_times"])
        print("SNR: ", data_augment_params["white_noise_snr"])
        print("STD: ", data_augment_params["white_noise_std"])
        for augment_index in range(data_augment_params["augment_times"]):
            white_noise = get_white_noise(X, data_augment_params["white_noise_std"],
                                          data_augment_params["white_noise_snr"])
            X_new = X.copy() + white_noise
            new_X_list.append(X_new)
            new_Y_list.append(Y)

    new_X = np.concatenate(new_X_list)
    new_Y = np.concatenate(new_Y_list)

    return new_X, new_Y


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, monitor="val_loss",
                 first_phase_loss=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            monitor (string): "val_accuracy" or "val_loss" to be tracked
            first_phase_loss (float): (Optional) The training loss used in two phase training
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Counter for epoch number without decreasing of monitoring value (EarlyStop triggered when this equals the value of patience)
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor
        self.val_monitor_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.first_phase_loss = first_phase_loss

    def __call__(self, val_monitor, model):
        if self.monitor == "val_acc":
            score = val_monitor
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_monitor, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_monitor, model)
                self.counter = 0
        else:
            score = -val_monitor

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_monitor, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            elif self.first_phase_loss is not None and score > -self.first_phase_loss:

                self.trace_func(f'EarlyStopping triggered because of lower val_loss than first phase loss')
                self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_monitor, model)
                self.counter = 0

    def save_checkpoint(self, val_monitor, model):
        '''Print message when validation loss decrease.'''
        if self.monitor == "val_accuracy":
            monitor_msg = "Validation accuracy"
        else:
            monitor_msg = "Validation loss"
        if self.verbose:
            self.trace_func(
                f'{monitor_msg} decreased ({self.val_monitor_min:.6f} --> {val_monitor:.6f}).  Early Stopping.')
        self.val_monitor_min = val_monitor


def train(model, X_train, Y_train, X_valid, Y_valid, loss_func, optimizer, n_epochs, batch_size, early_stop,
          early_stop__patience, early_stop__monitor, device, best_path="", best_model_path="", save_best_model=True,
          valid_size=0.2, checkpoint_monitor="val_loss", first_phase_loss=None):
    train_accuracy_list = []
    val_accuracy_list = []

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    valid_data = torch.utils.data.TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(Y_valid))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    best_loss = math.inf
    early_stop_obj = EarlyStopping(patience=early_stop__patience, verbose=True, path=best_path,
                                   monitor=early_stop__monitor, first_phase_loss=first_phase_loss)
    batch_num = math.ceil(X_train.shape[0] / batch_size)
    for i_epochs in range(n_epochs):

        model.train()
        train_loss = 0.0
        n_correct = 0
        for i_iter, (train_data, train_label) in enumerate(train_dataloader):
            train_data = train_data.to(device, dtype=torch.float)
            train_label = train_label.to(device, dtype=torch.long)
            optimizer.zero_grad()
            train_pred = model(train_data)
            batch_loss = loss_func(train_pred, train_label)
            n_correct += train_pred.max(dim=1)[1].eq(train_label).sum().item()
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader.dataset)
        train_acc = 100. * n_correct / len(train_dataloader.dataset)
        train_accuracy_list.append(train_acc)
        if valid_size > 0:
            print(f'epoch{i_epochs:>3d}  train_loss:{train_loss:.4f}  train_acc:{train_acc:.1f}%  ', end="")
        else:
            print(f'epoch{i_epochs:>3d}  train_loss:{train_loss:.4f}  train_acc:{train_acc:.1f}%  ')

        model.eval()
        val_loss = 0
        n_correct = 0
        with torch.no_grad():
            for i_iter, (valid_data, valid_label) in enumerate(valid_dataloader):
                valid_data = valid_data.to(device, dtype=torch.float)
                valid_label = valid_label.to(device, dtype=torch.long)
                valid_pred = model(valid_data)
                batch_loss = loss_func(valid_pred, valid_label)
                val_loss += batch_loss
                n_correct += valid_pred.max(dim=1)[1].eq(valid_label).sum().item()
        val_loss /= len(valid_dataloader.dataset)
        val_acc = 100. * n_correct / len(valid_dataloader.dataset)
        val_accuracy_list.append(val_acc)
        print(f'valid_loss:{val_loss:.4f}  valid_acc:{val_acc:.1f}%')

        if save_best_model:
            if checkpoint_monitor == "val_accuracy":
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_path)
                    print('best acc=%.5f, best model saved.' % best_acc)
            else:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), best_path)
                    print('best loss=%.5f, best model saved.' % float(best_loss))

        if early_stop:
            if early_stop__monitor == "val_loss":
                early_stop_obj(val_loss, model)
            else:
                early_stop_obj(val_acc, model)
            if early_stop_obj.early_stop:
                print("Early stopping triggered")
                break

    return model, train_accuracy_list, val_accuracy_list


def test(model, X_test, Y_test, loss_func, device, n_classes=None, pretrain=False, batch_size=1):
    model.eval()
    print(X_test.shape)
    print(Y_test.shape)
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    n_correct = 0
    test_loss = 0
    pred_list = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            data = data.to(device, dtype=torch.float)
            if pretrain:
                data.idx_in_dataloader = idx
            target = target.to(device, dtype=torch.long)
            predict = model(data)
            pred_list.append(predict)
            loss = loss_func(predict, target)
            test_loss += loss
            n_correct += predict.max(dim=1)[1].eq(target).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_score = 100. * n_correct / len(test_dataloader.dataset)
    return test_loss, test_score


def cross_validation(model, X_train, Y_train, stratified=False, K=5, use_deep_learning_models=False,
                     subject_session_name="", model_name="", info=None, valid_size=0.2, dataset=None,
                     params=None, data_augment_params=None, plot_params=None, seed=1):
    subject_name = subject_session_name.split("s")[0]
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []
    n_classes = len(np.unique(Y_train))

    if params["DL_Training"]["use_CUDA"]:
        model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=params[model_name]["learning_rate"])
    loss = nn.CrossEntropyLoss().cuda()

    cv_test = KFold(K, shuffle=True, random_state=seed)

    datetime_mark_init = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = "weights/" + model_name + "/within/subject" + subject_name + "/"
    mkdir(path)
    torch.save(model.state_dict(), path + "init_" + datetime_mark_init + ".pth")

    for test_split_index, (train_index, test_index) in enumerate(cv_test.split(X_train, Y_train)):

        X_test = X_train[test_index]
        Y_test = Y_train[test_index]

        X_train_set = X_train[train_index]
        Y_train_set = Y_train[train_index]

        print("test_split_index: ", test_split_index)
        datetime_mark = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        print(datetime_mark)

        n_valid = math.ceil(valid_size * X_train_set.shape[0])
        cv_valid = ShuffleSplit(test_size=n_valid, random_state=seed)

        X_train_train_idx, X_train_valid_idx = next(cv_valid.split(X_train_set, Y_train_set))
        X_train_train_set = X_train_set[X_train_train_idx]
        Y_train_train_set = Y_train_set[X_train_train_idx]
        X_train_valid_set = X_train_set[X_train_valid_idx]
        Y_train_valid_set = Y_train_set[X_train_valid_idx]

        print("\nsubject: %s of model: %s" % (subject_session_name, model_name))

        if data_augment_params["use_data_augment"]:
            dataset_name = dataset.name
            X_train_gen, Y_train_gen = data_augment(X_train_train_set, Y_train_train_set, subject_session_name,
                                                    dataset_name, data_augment_params,
                                                    train_index[X_train_train_idx], seed,
                                                    info=info[0])

            X_train_train_set = np.concatenate([X_train_train_set, X_train_gen])
            Y_train_train_set = np.concatenate([Y_train_train_set, Y_train_gen])

        model.load_state_dict(torch.load(
            "weights/" + model_name + "/within/subject" + subject_name + "/init_" + datetime_mark_init + ".pth"))

        best_path = "weights/" + model_name + "/within/subject" + subject_name + "/valid_size=" + str(
            valid_size) + "_best_" + datetime_mark + "_split" + str(test_split_index) + ".pth"

        training_time_start = time.time()
        early_stop = params["DL_Training"]["early_stop"]
        train(model, X_train_train_set, Y_train_train_set, X_train_valid_set,
              Y_train_valid_set, loss, opt, params["DL_Training"]["n_epochs"],
              params["DL_Training"]["batch_size"],
              early_stop=early_stop,
              early_stop__patience=params[model_name]["validation_stopping_patience"],
              early_stop__monitor=params["DL_Training"]["early_stop_monitor"],
              device=params["DL_Training"]["device"], best_path=best_path,
              save_best_model=True, valid_size=valid_size,
              checkpoint_monitor=params["DL_Training"]["checkpoint_monitor"])

        training_time_end = time.time()
        training_time = training_time_end - training_time_start
        print("Training time on subject %s: %s (s)." % (subject_session_name, training_time))

        model.load_state_dict(torch.load(best_path))

        model.load_state_dict(torch.load(best_path))
        _, train_score_elem = test(model, X_train_train_set, Y_train_train_set, loss,
                                   params["DL_Training"]["device"],
                                   n_classes=n_classes,
                                   batch_size=params["DL_Training"]["batch_size"])
        train_accuracy_list.append(train_score_elem)

        _, valid_score_elem = test(model, X_train_valid_set, Y_train_valid_set, loss,
                                   params["DL_Training"]["device"],
                                   n_classes=n_classes,
                                   batch_size=params["DL_Training"]["batch_size"])
        valid_accuracy_list.append(valid_score_elem)

        test_loss, test_score = test(model, X_test, Y_test, loss,
                                     params["DL_Training"]["device"],
                                     n_classes=n_classes,
                                     batch_size=params["DL_Training"]["batch_size"])
        test_accuracy_list.append(test_score)

    train_accuracy = np.array(train_accuracy_list).mean()
    valid_accuracy = np.array(valid_accuracy_list).mean()
    test_accuracy = np.array(test_accuracy_list).mean()

    print("Mean train score of subject %s of model %s: %.4f" % (subject_session_name, model_name, train_accuracy))
    print("Mean test score of subject %s of model %s: %.4f" % (subject_session_name, model_name, test_accuracy))
    del model
    gc.collect()  # Force the GarbageCollector to release unused memory
    return train_accuracy, valid_accuracy, test_accuracy


def test_on_each_subject(dataset, dir_datetime_mark=None, datetime_mark=None):
    print("\n###############################################################################\n")
    datetime_mark = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(datetime_mark)
    print("Device Name: ", torch.cuda.get_device_name(0) if use_CUDA else "CPU")
    print("Seed: ", seed)
    print("Dataset: ", dataset_name)
    print("Class of intereset: ", class_of_interest)
    print("Data type: ", data_type)
    print("Number of subjects: ", n_subjects)
    if n_subjects == 1:
        print("Target subject: ", target_subject)
    else:
        print("Start subject: ", start_subject)
        print("End subject: ", end_subject)
    print("Tmin: ", str(dataset.tmin))
    print("Tmax: ", str(dataset.tmax))
    model_used = []
    for key in use_models:
        if use_models[key]:
            model_used.append(key)
    print("Model used:", str(model_used))
    print("Scenario: ", scenario_name)
    if "cross_validation" in scenario_name:
        print("K: ", K)

    print("------------------------------------------------------------------")
    print("Preprocessing settings: \n")
    print("Use CAR: ", preprocessing_params["use_CAR"])
    print("Use moving_average_std: ", preprocessing_params["moving_average_std"])
    print("Use band-pass filtering: ", preprocessing_params["use_band_pass"])
    if preprocessing_params["use_band_pass"]:
        for item in band_pass_params:
            print(item + ": " + str(band_pass_params[item]))
    print("------------------------------------------------------------------")
    if data_augment_params["use_data_augment"]:
        print("Data Augmentation settings: \n")
        print("Method: ", data_augment_params["method"])
        print("Augment times: ", data_augment_params["augment_times"])
        if "noise" in data_augment_params["method"]:
            for item in noise_adding_params:
                print(item + ": " + str(noise_adding_params[item]))
    else:
        print("Use Data Augmentation: False")
    print("------------------------------------------------------------------")

    model_names = []
    model_index = dict()
    index = 0

    for model in use_models:
        if use_models[model]:
            model_names.append(model)
            model_index[model] = index
            index += 1

    print("Model settings: ")
    for model_name in model_names:
        print("\nParameter Setting of model: ", model_name)
        for param_name in params[model_name]:
            if param_name != "datetime_marks_per_subject":
                print(param_name + ": " + str(params[model_name][param_name]))

    print("------------------------------------------------------------------")
    print("Training settings: \n")
    for item in training_params:
        print(item + ": " + str(training_params[item]))
    print("valid size: ", valid_size)
    print("------------------------------------------------------------------")

    n_sessions_per_subject = dataset.n_sessions_per_subject
    train_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))
    valid_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))
    test_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))

    subject_session_names = []
    for subject_idx in range(start_subject - 1, end_subject):
        print("\n###############################################################################\n")
        if n_subjects == 1:
            subject_idx = target_subject - 1
        subject_name = str(subject_idx + 1)

        n_sessions = dataset.n_sessions_per_subject if dataset.n_sessions_per_subject is not None else 1
        for session_idx in range(n_sessions):

            subject_session_name = subject_name + "s" + str(session_idx + 1)
            if not dataset.use_all_subject_sessions \
                    and dataset.subject_session_names_included is not None \
                    and subject_session_name not in dataset.subject_session_names_included:
                print("Skip subject%s.\n" % subject_session_name)
                continue
            subject_session_names.append(subject_session_name)

            print("Loading data of subject %s... " % subject_session_name)

            dataset.load([subject_idx + 1], session_list=[session_idx + 1], data_type=data_type)
            print("Loading data of subject %s... Done." % subject_session_name)

            Y_train_list = []
            for model_name in model_names:

                X_train_list = []
                info_list = []
                Y_train = None

                if "EEG" in data_type:

                    raw_file_train = dataset.raw_file_train_list[0]
                    raw_file_test = None

                    print("Preprocessing data of subject %s... " % subject_session_name)

                    if type(raw_file_train) != list:
                        raw_file_train = [raw_file_train]

                    X_train = []
                    Y_train = []
                    for raw_file_idx, raw_file in enumerate(raw_file_train):
                        X_train_elem, Y_train_elem, info, frequency_bands_list = preprocessing_func(raw_file,
                                                                                                    dataset,
                                                                                                    model_name,
                                                                                                    preprocessing_params,
                                                                                                    load_test_label=False,
                                                                                                    subject_idx=subject_idx,
                                                                                                    session_idx=raw_file_idx,
                                                                                                    seed=seed)

                        X_train.append(X_train_elem)
                        Y_train.append(Y_train_elem)
                    X_train = np.concatenate(X_train)
                    Y_train = np.concatenate(Y_train)

                    X_train_list.append(X_train)
                    info_list.append(info)
                    X_test = None
                    Y_test = None

                assert (len(X_train_list) > 0)

                X_train = X_train_list if len(X_train_list) > 1 else X_train_list[0]

                print("Preprocessing data of subject %s... Done." % subject_session_name)

                Y_train_list.append(Y_train)

                print("Building %s model of subject %s with scenario %s..." % (
                    model_name, subject_session_name, scenario_name))

                n_times = X_train.shape[-1]
                n_electrodes = X_train.shape[-2]
                n_classes = len(np.unique(Y_train))

                conv1_size = params["ShallowConvNet"]["conv1_size"]
                avg_pool_size = params["ShallowConvNet"]["avg_pool_size"]
                avg_pool_stride = params["ShallowConvNet"]["avg_pool_stride"]
                in_features_length = int((n_times - conv1_size + 1 - avg_pool_size) // avg_pool_stride + 1)
                model = ShallowConvNet(n_channels=n_electrodes,
                                       conv1_size=params["ShallowConvNet"]["conv1_size"],
                                       avg_pool_size=params["ShallowConvNet"]["avg_pool_size"],
                                       avg_pool_stride=params["ShallowConvNet"]["avg_pool_stride"],
                                       in_features_length=in_features_length,
                                       n_classes=n_classes,
                                       max_norm_temporal_conv=params["ShallowConvNet"][
                                           "max_norm_temporal_conv"],
                                       max_norm_spatial_conv=params["ShallowConvNet"][
                                           "max_norm_spatial_conv"],
                                       max_norm_linear=params["ShallowConvNet"]["max_norm_linear"],
                                       dropout_rate=params["ShallowConvNet"]["dropout_rate"])

                train_acc_elem, valid_acc_elem, test_acc_elem = cross_validation(model, X_train, Y_train,
                                                                                 stratified=False, K=K,
                                                                                 subject_session_name=subject_session_name,
                                                                                 model_name=model_name, info=info_list,
                                                                                 valid_size=valid_size, dataset=dataset,
                                                                                 params=params,
                                                                                 data_augment_params=data_augment_params,
                                                                                 plot_params=plot_params,
                                                                                 seed=seed)

                if n_subjects == 1:
                    subject_idx = 0
                subject_session_idx = subject_idx * n_sessions_per_subject + session_idx
                train_acc_array[subject_session_idx][model_index[model_name]] = train_acc_elem
                valid_acc_array[subject_session_idx][model_index[model_name]] = valid_acc_elem
                test_acc_array[subject_session_idx][model_index[model_name]] = test_acc_elem
                print("Building %s model of subject %s with scenario %s. Done." % (
                    model_name, subject_session_name, scenario_name))

    train_acc_array[-1] = train_acc_array[:-1].mean(axis=0)
    if valid_size > 0:
        valid_acc_array[-1] = valid_acc_array[:-1].mean(axis=0)
    test_acc_array[-1] = test_acc_array[:-1].mean(axis=0)
    mean_test_acc = test_acc_array[-1]

    print("\ntrain_acc_array:", train_acc_array)
    print("\nvalid_acc_array:", valid_acc_array)
    print("\ntest_acc_array:", test_acc_array)
    print("\nMean test score: ", mean_test_acc)

    result_train = []
    if valid_size > 0:
        result_valid = []
    result_test_acc = []

    subject_names = subject_session_names
    subject_names.append("Average")

    for i in range(len(test_acc_array)):
        result_train.append([])
        if valid_size > 0:
            result_valid.append([])
        result_test_acc.append([])

        for j in range(len(test_acc_array[i])):
            result_train[i].append("%.4f" % train_acc_array[i][j])
            if valid_size > 0:
                result_valid[i].append("%.4f" % valid_acc_array[i][j])
            result_test_acc[i].append("%.4f" % test_acc_array[i][j])

    pdf_result_train = pd.DataFrame(result_train, index=subject_names, columns=model_names)
    if valid_size > 0:
        pdf_result_valid = pd.DataFrame(result_valid, index=subject_names, columns=model_names)
    pdf_result_test_acc = pd.DataFrame(result_test_acc, index=subject_names, columns=model_names)

    if dir_datetime_mark is None:
        dir_datetime_mark = datetime_mark
    dir_path = "results/" + dir_datetime_mark
    if not os.path.exists(dir_path):
        mkdir(dir_path)
    pdf_result_train.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__train.csv")
    if valid_size > 0:
        pdf_result_valid.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__valid.csv")
    pdf_result_test_acc.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__test_acc.csv")

    return dir_datetime_mark


if __name__ == "__main__":
    start_time = time.time()
    print("\n###############################################################################\n")
    print("Run one time")
    test_on_each_subject(dataset)
    end_time = time.time()
    time_cost = end_time - start_time
    print("\nRun time: %f (s)." % time_cost)
