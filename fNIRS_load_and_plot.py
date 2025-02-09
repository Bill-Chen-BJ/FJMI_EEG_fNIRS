# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory
import mne
from matplotlib.ticker import FormatStrFormatter
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_cnt, read_raw_nirx

import numpy as np
import matplotlib.pyplot as plt

import pickle
import gc
import os.path

dataset_name = "FineMI"

if dataset_name == "FineMI":
    tmin = -4.
    tmax = 14.


if dataset_name == "FineMI":
    n_subjects = 18

start_subject = 1

if dataset_name == "FineMI":
    sample_rate_dataset = 1000

# Preprocessing Settings
filter_order_fnirs = 6
filter_type_fnirs = "butter"

# Package Global Settings
seed = 1
mne.set_log_level(verbose="WARNING")
n_jobs = 2


plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plot_all_subjects = True
# load_mean_epochs = True
load_mean_epochs = False
plot_mean_of_selected_channels = True
plot_by_class = True
plot_trends = True
plot_topo = True
channel_names = [{"S": 1, "D": 1},
                 {"S": 1, "D": 2},
                 {"S": 1, "D": 5},
                 {"S": 2, "D": 1},
                 {"S": 2, "D": 2},
                 {"S": 2, "D": 3},
                 {"S": 3, "D": 2},
                 {"S": 3, "D": 3},
                 {"S": 3, "D": 4},
                 {"S": 3, "D": 7},
                 {"S": 4, "D": 3},
                 {"S": 4, "D": 4},
                 {"S": 5, "D": 5},
                 {"S": 5, "D": 6},
                 {"S": 6, "D": 2},
                 {"S": 6, "D": 5},
                 {"S": 6, "D": 6},
                 {"S": 6, "D": 7},
                 {"S": 7, "D": 6},
                 {"S": 7, "D": 7},
                 {"S": 7, "D": 8},
                 {"S": 8, "D": 4},
                 {"S": 8, "D": 7},
                 {"S": 8, "D": 8}, ]
source_names = ['FCC1h', 'FFC3h', 'FCC5h', 'FFT7h', 'CPP1h', 'CCP3h', 'CPP5h', 'TTP7h']
detector_names = ['FFC1h', 'FCC3h', 'FFC5h', 'FTT7h', 'CCP1h', 'CPP3h', 'CCP5h', 'TPP7h']


def load_raw_fnirs_data(subject_name, dataset_name):

    if dataset_name == "FineMI":
        n_blocks = 8
        raw_file_train_list = []

        print("Subject: ", subject_name)
        if subject_name == '1':
            # subject 1 session 1
            # Block1-4
            file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block1-4"
            raw_file = read_raw_nirx(file_name, preload=True)
            idx_to_remove = np.arange(-40, 0)
            raw_file.annotations.delete(idx_to_remove)
            raw_file.crop_by_annotations()
            raw_file_train_list.append(raw_file)

            # Block5-8
            for block_idx in range(4, n_blocks):
                file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
                raw_file = read_raw_nirx(file_name, preload=True)
                raw_file_train_list.append(raw_file)
        else:
            # other sessions
            for block_idx in range(n_blocks):
                file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
                raw_file = read_raw_nirx(file_name, preload=True)
                if subject_name == "5" and block_idx == 5:
                    raw_file.annotations.delete(0)
                    raw_file.crop_by_annotations()
                raw_file_train_list.append(raw_file)

        raw_file_train = concatenate_raws(raw_file_train_list)
        raw_file_test = None
    else:
        print("Unknown Dataset!")
    return raw_file_train, raw_file_test


def extract_epoch(raw_file, event_id, picks, tmin, tmax):
    events, _ = events_from_annotations(raw_file, event_id=event_id)
    epochs = Epochs(raw_file, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=(-3.9, -2.), preload=True, verbose=False)
    return epochs


def get_data_from_raw_file(raw_file_train, raw_file_test, subject, dataset_name, picks_train, picks_test, tmin, tmax,
                           data_type="fNIRS"):

    if dataset_name == "FineMI":
        event_id_train = {
            "1.0": 1,
            "2.0": 2,
            "3.0": 3,
            "4.0": 4,
            "5.0": 5,
            "6.0": 6,
            "7.0": 7,
            "8.0": 8,
        }

    # extract data
    if not type(raw_file_train) == list:
        epochs_train = extract_epoch(raw_file_train, event_id_train, picks_train, tmin, tmax)
        epoch_info = epochs_train.info
    else:
        epochs_train = [extract_epoch(raw_file_train_elem, event_id_train, picks_train, tmin, tmax) for
                        raw_file_train_elem in raw_file_train]
        epoch_info = epochs_train[0].info

    if not type(epochs_train) == list:
        train_data = epochs_train.get_data()
    else:
        train_data = np.concatenate(
            [epochs_train_elem.get_data() for epochs_train_elem in epochs_train])

    del raw_file_train
    gc.collect()

    # save epochs
    if not os.path.exists("tmp"):
        os.makedirs('tmp')
    epochs_train.save("tmp/subject" + str(subject_session_name) + "_" + data_type + "-epo.fif", overwrite=True)

    # get labels
    if not type(epochs_train) == list:
        train_labels = epochs_train.events[:, -1] - 1
    else:
        train_labels = np.concatenate([epochs_train_elem.events[:, -1] - 1 for epochs_train_elem in epochs_train])

    del epochs_train
    gc.collect()
    return train_data, train_labels, None, None, epoch_info


##############################################################################

def preprocessing_fnirs(raw_file_train, raw_file_test, subject_session_name, dataset_name, tmin, tmax):
    raw_file_train_od = mne.preprocessing.nirs.optical_density(raw_file_train)

    raw_file_train_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_file_train_od)

    iir_params = {
        "output": "sos",
        "order": filter_order_fnirs,
        "ftype": filter_type_fnirs
    }

    raw_file_train_haemo = raw_file_train_haemo.filter(0.01, 0.1, method='iir', iir_params=iir_params,
                                                       n_jobs=n_jobs)


    X_train, Y_train, X_test, Y_test, epoch_info = get_data_from_raw_file(raw_file_train_haemo, None,
                                                                          subject_session_name,
                                                                          dataset_name,
                                                                          None, None,
                                                                          tmin,
                                                                          tmax, data_type="fNIRS")
    frequency_list = []
    return X_train, Y_train, X_test, Y_test, frequency_list, epoch_info


def plot_fNIRS(epochs, subject, plot_by_class=False):
    tasks = [
        {
            "joint": "Hand",
            "move": "flexion/extension"
        },
        {
            "joint": "Wrist",
            "move": "flexion/extension"
        },
        {
            "joint": "Wrist",
            "move": "adduction/abduction"
        },
        {
            "joint": "Elbow",
            "move": "pronation/supination"
        },
        {
            "joint": "Elbow",
            "move": "flexion/extension"
        },
        {
            "joint": "Shoulder",
            "move": "pronation/supination"
        },
        {
            "joint": "Shoulder",
            "move": "adduction/abduction"
        },
        {
            "joint": "Shoulder",
            "move": "flexion/extension"
        }
    ]
    n_classes = len(tasks)
    dir_path = "img/fnirs/subject" + subject
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created directory: {dir_path}')

    mean_epoch_dict_list = []
    for c in range(n_classes):
        filename = dir_path + "/" + str(c) + ".pkl"
        if load_mean_epochs:
            f_read = open(filename, 'rb')
            mean_epoch_dict = pickle.load(f_read)
            f_read.close()
        else:
            epochs_list = []
            for subject_name in range(1, 19):
                epochs_elem = mne.read_epochs("tmp/subject" + str(subject_name) + "_" + data_type + "-epo.fif",
                                              preload=True)

                epochs_list.append(epochs_elem[str(c + 1) + ".0"].copy())
                del epochs_elem
                gc.collect()
            epochs_all = mne.concatenate_epochs(epochs_list)
            mean_epoch_dict = {'HbO': epochs_all.average(picks='hbo'), 'HbR': epochs_all.average(picks='hbr')}
            for condition in mean_epoch_dict:
                mean_epoch_dict[condition].rename_channels(lambda x: x[:-4])
            f_save = open(filename, 'wb')
            pickle.dump(mean_epoch_dict, f_save)
            f_save.close()
            del epochs_list
            del epochs_all
            gc.collect()
        mean_epoch_dict_list.append(mean_epoch_dict)

    if plot_trends:
        for c in range(n_classes):

            color_dict = dict(HbO='#AA3377', HbR='b')

            selected_fNIRS_channels = ["CCP3h-FCC3h", "CCP3h-CCP5h", "FCC5h-CCP5h", "FCC5h-FCC3h"]
            picks_fNIRS = []
            for selected_fNIRS_channel in selected_fNIRS_channels:
                S = selected_fNIRS_channel.split("-")[0]
                D = selected_fNIRS_channel.split("-")[1]
                S_idx = source_names.index(S)
                D_idx = detector_names.index(D)
                picks_fNIRS.append("S" + str(S_idx + 1) + "_" + "D" + str(D_idx + 1))

            fig_trend, axes_trend = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

            mne.viz.plot_compare_evokeds(mean_epoch_dict_list[c], combine="mean", picks=picks_fNIRS,
                                         axes=axes_trend, show=False,
                                         truncate_xaxis=False, truncate_yaxis=False, legend="lower right",
                                         ylim=dict(hbo=[-0.05, 0.15], hbr=[-0.05, 0.15]),
                                         colors=color_dict,
                                         title=tasks[c]["joint"] + " " + tasks[c]["move"])

            axes_trend.set_title(tasks[c]["joint"] + " " + tasks[c]["move"],
                                 fontdict={'fontsize': 20, 'fontweight': 'bold'})
            axes_trend.legend(["HbO", "HbR"], fontsize=24, loc="upper right")
            legend = axes_trend.get_legend()
            for text in legend.get_texts():
                text.set_fontweight('extra bold')
            for line in legend.get_lines():
                line.set_linewidth(4.0)

            axes_trend.tick_params(labelsize=24)
            axes_trend.set_xlabel("Times (s)", fontsize=24, fontweight='bold')
            axes_trend.set_ylabel("µM", fontsize=24, fontweight='bold')
            for line in axes_trend.get_lines():
                line.set_linewidth(4.0)

            for tick in axes_trend.get_xticklabels():
                tick.set_fontweight('bold')

            for tick in axes_trend.get_yticklabels():
                tick.set_fontweight('bold')

            plt.tight_layout()
            fig_trend.show()
            fig_trend.savefig("img/fnirs/subject" + subject + "/trend_4_channel_around_C3_" + str(c) + ".png",
                              dpi=300)

    if plot_topo:
        for c in range(n_classes):
            times = 9.  # 8-10s
            half_time = 1.
            topomap_args = dict(extrapolate='local', cmap="RdBu_r")
            mean_epoch = mean_epoch_dict_list[c]
            vlim_hbo = (-0.2, 0.2)

            fig_topo_hbo, axes_topo_hbo = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            mean_epoch["HbO"].plot_topomap(ch_type='hbo', times=times, colorbar=True,
                                           axes=axes_topo_hbo,
                                           average=half_time,
                                           vlim=vlim_hbo, **topomap_args)

            cbar1 = axes_topo_hbo[0].images[0].colorbar

            formatter = FormatStrFormatter('%.2f')
            cbar1.ax.yaxis.set_major_formatter(formatter)
            cbar1.ax.title.set_size(20)
            cbar1.ax.tick_params(labelsize=20)
            axes_topo_hbo[0].set_title(tasks[c]["joint"] + " " + tasks[c]["move"] + "(HbO)", fontdict={'fontsize': 20})
            fig_topo_hbo.tight_layout()

            ll, bb, ww, hh = axes_topo_hbo[-1].get_position().bounds
            axes_topo_hbo[-1].set_position([ll, bb, ww * 0.05, hh])

            plt.tight_layout()
            fig_topo_hbo.savefig("img/fnirs/subject" + subject + "/topo_hbo_" + str(c) + ".png", dpi=300)

    return None


if __name__ == "__main__":
    data_type = "fNIRS"
    for subject in range(start_subject, 19):
        subject_session_name = str(subject)

        raw_file_train, raw_file_test = load_raw_fnirs_data(subject_session_name,
                                                            dataset_name)


        X_train, Y_train, X_test, Y_test, frequency_list, epoch_info = preprocessing_fnirs(raw_file_train,
                                                                                           raw_file_test,
                                                                                           subject_session_name,
                                                                                           dataset_name, tmin,
                                                                                           tmax)

        print("\nLoading data of subject %d...  done.\n" % (subject + 1))

    plot_fNIRS(None, "All", plot_by_class)
