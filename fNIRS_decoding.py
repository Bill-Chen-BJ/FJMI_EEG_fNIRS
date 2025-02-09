# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory
import random
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from numpy.polynomial import Polynomial

from mne.io import concatenate_raws, read_raw_cnt, read_raw_nirx
from mne.channels import read_custom_montage
from mne import Epochs, events_from_annotations, pick_types

import datetime
import time
import math
import os
import gc

dataset_name = "FineMI"
class_of_interest = ["1", "7"]
data_type = ["fNIRS"]
use_hbr = True

preprocessing_params = {
    "use_band_pass": True,
    "filter_order_fnirs": 6,
    "filter_type_fnirs": "butter",
    "lower_bound_fnirs": 0.01,
    "upper_bound_fnirs": 0.1,
    "moving_average_std_fnirs": True,
    "n_jobs": 2
}

use_models = {
    "ML:fNIRS_Union:Mean;Slope+Zscore+SVM": True,
}

params = {
    "SVM": {
        "kernel": "linear",
        "svm_C": 1
    }
}

one_time = True

# Scenario
scenario_name = "intra_subject_cross_validation"

K = 5
valid_size = 0.2

plot_params = {}

seed = 1
n_jobs = 4
mne.set_log_level(verbose="WARNING")
np.random.seed(seed)
random.seed(seed)


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


dataset = FineMI(tmin=3, tmax=7, baseline_tmin=-4, baseline_tmax=-2, class_of_interest=class_of_interest)

start_subject = 1
n_subjects = 18
end_subject = n_subjects
target_subject = 3


def extract_epoch(raw_file, events, event_id, tmin, tmax, sample_rate=1000, down_sample=False,
                  down_sample_rate=1000, n_jobs=1, picks=None, baseline_tmin=None, baseline_tmax=None):
    if baseline_tmin is None and baseline_tmax is None:
        baseline = None
    else:
        baseline = (baseline_tmin, baseline_tmax)
    epochs = Epochs(raw_file, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, verbose=False)

    if down_sample and down_sample_rate != sample_rate:
        epochs.resample(down_sample_rate, npad='auto', n_jobs=n_jobs)

    return epochs


def get_epochs_data(raw_file, dataset, event_id, picks=None, data_type="fNIRS"):
    events, _ = events_from_annotations(raw_file, event_id=event_id)
    tmin = dataset.tmin
    tmax = dataset.tmax
    baseline_tmin = None
    baseline_tmax = None

    epochs = extract_epoch(raw_file, events, event_id, tmin, tmax, dataset.sample_rate_fnirs,
                           dataset.resample_fnirs, dataset.resample_rate_fnirs, picks=picks,
                           baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax)
    data = epochs.get_data()
    label = epochs.events[:, -1]

    label_names = np.unique(label)
    for label_idx, label_name in enumerate(label_names):
        label[label == label_name] = label_idx

    # clean last few points
    n_samples = data.shape[-1]
    if data_type == "fNIRS":
        if dataset.resample_fnirs:
            if (n_samples - (dataset.tmax - dataset.tmin) * dataset.resample_rate_fnirs) > 0:
                remainder = int(n_samples - (dataset.tmax - dataset.tmin) * dataset.resample_rate_fnirs)
                if remainder != 0:
                    data = data[:, :, :-remainder]
        elif (n_samples - (dataset.tmax - dataset.tmin) * dataset.sample_rate_fnirs) > 0:
            remainder = int(n_samples - (dataset.tmax - dataset.tmin) * dataset.sample_rate_fnirs)
            if remainder != 0:
                data = data[:, :, :-remainder]

    return data, label, epochs.info


def exponential_moving_standardize(data, factor_new=0.001, eps=1e-4):
    """
    :param data: (n_channels, n_times)
    :param factor_new:
    :param eps:
    :return:
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    return standardized.T


def preprocessing_fnirs_func(raw_file_fnirs, dataset, preprocessing_params, subject_idx=0):
    raw_file_fnirs_od = mne.preprocessing.nirs.optical_density(raw_file_fnirs)

    raw_file_fnirs_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_file_fnirs_od)

    if preprocessing_params["use_band_pass"]:
        iir_params = {
            "output": "sos",
            "order": preprocessing_params["filter_order_fnirs"],
            "ftype": preprocessing_params["filter_type_fnirs"]
        }

        raw_file_fnirs_haemo = raw_file_fnirs_haemo.filter(preprocessing_params["lower_bound_fnirs"],
                                                           preprocessing_params["upper_bound_fnirs"],
                                                           method='iir',
                                                           iir_params=iir_params,
                                                           n_jobs=preprocessing_params["n_jobs"])

    if preprocessing_params["moving_average_std_fnirs"]:
        raw_file_fnirs_haemo = raw_file_fnirs_haemo.apply_function(exponential_moving_standardize,
                                                                   n_jobs=preprocessing_params["n_jobs"],
                                                                   channel_wise=False)

    X_fnirs, Y_fnirs, info_fnirs = get_epochs_data(raw_file_fnirs_haemo, dataset,
                                                   dataset.event_id_fnirs,
                                                   data_type="fNIRS")

    frequency_bands_list = [
        {
            "fmin": preprocessing_params["lower_bound_fnirs"],
            "fmax": preprocessing_params["upper_bound_fnirs"]
        }
    ]
    return X_fnirs, Y_fnirs, info_fnirs, frequency_bands_list


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


class Mean(TransformerMixin, BaseEstimator):

    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return np.mean(X, axis=self.axis)


class Slope(TransformerMixin, BaseEstimator):

    def __init__(self, n_len, method="interval"):
        self.n_len = n_len
        self.method = method

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        n_trials, n_channels, n_times = X.shape
        slopes_of_trials = []

        if self.method == "interval":
            slopes_of_trials = (X[:, :, n_times - 1] - X[:, :, 0]) / n_times
        else:
            for trial_idx in range(n_trials):
                slopes = []
                for channel_idx in range(n_channels):
                    times = np.arange(self.n_len)
                    slope, intercept = Polynomial.fit(times, X[trial_idx, channel_idx], 1)
                    slopes.append(slope)
                slopes_of_trials.append(slopes)
        return np.array(slopes_of_trials)


def cross_validation(model, X_train, Y_train, K=5, subject_session_name="", model_name="", valid_size=0.2, seed=1):
    subject_name = subject_session_name.split("s")[0]
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []
    n_classes = len(np.unique(Y_train))

    cv_test = KFold(K, shuffle=True, random_state=seed)

    for test_split_index, (train_index, test_index) in enumerate(cv_test.split(X_train, Y_train)):  # Divide into K folds
        # Pick 1 from K folds as the test set
        X_test = X_train[test_index]
        Y_test = Y_train[test_index]

        # Train the model on data from other K-1 folds
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

        print("Training set data shape: ", X_train_train_set.shape)
        model.fit(X_train_train_set, Y_train_train_set)
        train_accuracy_list.append(model.score(X_train_train_set, Y_train_train_set))
        if valid_size > 0:
            valid_accuracy_list.append(model.score(X_train_valid_set, Y_train_valid_set))
        else:
            valid_accuracy_list.append(0)

        test_accuracy_elem = model.score(X_test, Y_test)
        test_accuracy_list.append(test_accuracy_elem)

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
    print("Use band-pass filtering: ", preprocessing_params["use_band_pass"])

    model_names = []
    model_index = dict()
    index = 0

    for model in use_models:
        if use_models[model]:
            model_names.append(model)
            model_index[model] = index
            index += 1

    print("------------------------------------------------------------------")
    print("valid size: ", valid_size)
    print("------------------------------------------------------------------")

    n_sessions_per_subject = dataset.n_sessions_per_subject
    train_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the training set. Dimension: n_subjects* n_sessions_in_1_subject + 1 (Average)
    valid_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the validation set. Dimension: n_subjects* n_sessions_in_1_subject + 1 (Average)
    test_acc_array = np.zeros((n_subjects * n_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the test set. Dimension: n_subjects* n_sessions_in_1_subject + 1 (Average)

    subject_session_names = []

    for subject_idx in range(start_subject - 1, end_subject):
        print("\n###############################################################################\n")
        if n_subjects == 1:  # Run on 1 specified subject
            subject_idx = target_subject - 1
        subject_name = str(subject_idx + 1)

        n_sessions = dataset.n_sessions_per_subject if dataset.n_sessions_per_subject is not None else 1
        for session_idx in range(n_sessions):

            subject_session_name = subject_name + "s" + str(session_idx + 1)
            subject_session_names.append(subject_session_name)  # Record the subjenct number and session number

            print("Loading data of subject %s... " % subject_session_name)

            dataset.load([subject_idx + 1], session_list=[session_idx + 1], data_type=data_type)
            print("Loading data of subject %s... Done." % subject_session_name)

            Y_train_list = []
            for model_name in model_names:
                X_train_list = []
                info_list = []

                raw_file_fnirs = dataset.raw_file_fnirs_list[0]

                if type(raw_file_fnirs) != list:
                    raw_file_fnirs = [raw_file_fnirs]
                X_fnirs_train = []
                Y_fnirs_train = []
                info_fnirs = []
                for raw_file_idx, raw_file in enumerate(raw_file_fnirs):
                    X_fnirs_train_elem, Y_fnirs_train_elem, info_fnirs, frequency_bands_list = preprocessing_fnirs_func(
                        raw_file,
                        dataset,
                        preprocessing_params,
                        subject_idx=subject_idx
                    )

                    X_hbo_train_elem = X_fnirs_train_elem[:, ::2, :].copy()
                    X_hbr_train_elem = X_fnirs_train_elem[:, 1::2, :].copy()

                    if use_hbr:
                        X_fnirs_train_elem = np.concatenate([X_hbo_train_elem, X_hbr_train_elem], axis=1)
                    else:
                        X_fnirs_train_elem = X_hbo_train_elem
                    X_fnirs_train.append(X_fnirs_train_elem)
                    Y_fnirs_train.append(Y_fnirs_train_elem)
                X_fnirs_train = np.concatenate(X_fnirs_train)
                Y_fnirs_train = np.concatenate(Y_fnirs_train)

                X_train_list.append(X_fnirs_train)
                info_list.append(info_fnirs)
                Y_train = Y_fnirs_train.copy()

                assert (len(X_train_list) > 0)

                X_train = X_train_list if len(X_train_list) > 1 else X_train_list[0]

                print("Preprocessing data of subject %s... Done." % subject_session_name)

                Y_train_list.append(Y_train)

                print("Building %s model of subject %s with scenario %s..." % (
                    model_name, subject_session_name, scenario_name))

                n_times = X_train.shape[-1]
                mean = Mean(axis=-1)
                slope = Slope(n_times)
                ss = StandardScaler()
                svm = SVC(kernel=params["SVM"]["kernel"], C=params["SVM"]["svm_C"])
                union = FeatureUnion([("mean", mean), ("slope", slope)])
                model = Pipeline([("feature_union", union), ("standard_scale", ss), ('svm', svm)])
                train_acc_elem, valid_acc_elem, test_acc_elem = cross_validation(model, X_train, Y_train, K=K,
                                                                                 subject_session_name=subject_session_name,
                                                                                 model_name=model_name,
                                                                                 valid_size=valid_size,
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
