import glob
import re
import os
import pandas as pd
import numpy as np

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

import data_pre_processing

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{10.1145/3448112,
  author = {Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  title = {SelfHAR: Improving Human Activity Recognition through Self-Training with Unlabeled Data},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {5},
  number = {1},
  url = {https://doi.org/10.1145/3448112},
  doi = {10.1145/3448112},
  abstract = {Machine learning and deep learning have shown great promise in mobile sensing applications, including Human Activity Recognition. However, the performance of such models in real-world settings largely depends on the availability of large datasets that captures diverse behaviors. Recently, studies in computer vision and natural language processing have shown that leveraging massive amounts of unlabeled data enables performance on par with state-of-the-art supervised models.In this work, we present SelfHAR, a semi-supervised model that effectively learns to leverage unlabeled mobile sensing datasets to complement small labeled datasets. Our approach combines teacher-student self-training, which distills the knowledge of unlabeled and labeled datasets while allowing for data augmentation, and multi-task self-supervision, which learns robust signal-level representations by predicting distorted versions of the input.We evaluated SelfHAR on various HAR datasets and showed state-of-the-art performance over supervised and previous semi-supervised approaches, with up to 12% increase in F1 score using the same number of model parameters at inference. Furthermore, SelfHAR is data-efficient, reaching similar performance using up to 10 times less labeled data compared to supervised approaches. Our work not only achieves state-of-the-art performance in a diverse set of HAR datasets, but also sheds light on how pre-training tasks may affect downstream performance.},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  month = mar,
  articleno = {36},
  numpages = {30},
  keywords = {semi-supervised training, human activity recognition, unlabeled data, self-supervised training, self-training, deep learning}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk

Copyright (C) 2021 C. I. Tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:

        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """
    all_data = []
    all_labels = []
    all_user_ids = []

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)

        # Loop through files for every user of the trial
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how="any", inplace=True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                all_data.append(values)
                all_labels.append(np.repeat(label, values.shape[0]))
                all_user_ids.append(np.repeat(user_id, values.shape[0]))

            else:
                print("[ERR] User id not found", trial_user_file)

    # Concatenate all data
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    all_user_ids = np.concatenate(all_user_ids)

    # Normalize the entire dataset
    normalized_data = data_pre_processing.normalise_data(all_data)

    # Split the normalized data back into user-specific datasets
    for user_id in np.unique(all_user_ids):
        user_indices = np.where(all_user_ids == user_id)[0]
        user_data = all_data[user_indices]
        user_labels = all_labels[user_indices]
        if user_id not in user_datasets:
            user_datasets[user_id] = []
        user_datasets[user_id].append((user_data, user_labels))

    print("MotionSense user IDs:", list(user_datasets.keys()))
    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition

    Parameters:
        data_folder_path (str): Path to the folder containing the HHAR dataset.

    Returns:
        user_datasets (dict of {user_id: [(sensor_values, [activity_label, sensor_type])]}):

    """
    har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    har_dataset.dropna(how="any", inplace=True)
    har_dataset = har_dataset[["x", "y", "z", "gt", "User", "Device"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    har_users = har_dataset["user-id"].unique()

    all_data = []
    all_labels = []
    all_user_ids = []

    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        devices = user_extract["device"].values

        for i in range(len(data)):
            all_data.append(data[i])
            all_labels.append([labels[i], devices[i]])
            all_user_ids.append(user)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_user_ids = np.array(all_user_ids)

    # Normalize the entire dataset
    normalized_data = data_pre_processing.normalise_data(all_data)

    user_datasets = {}
    for user_id in np.unique(all_user_ids):
        user_indices = np.where(all_user_ids == user_id)[0]
        user_data = normalized_data[user_indices]
        user_labels = all_labels[user_indices]
        user_datasets[user_id] = [(user_data, user_labels)]

    print("HHAR user IDs:", list(user_datasets.keys()))
    return user_datasets



def process_wisdm_accelerometer_files(file_path):
    """
    Preprocess the accelerometer files of the WISDM dataset into the 'user-list' format.

    Parameters:
        file_path (str): Path to the WISDM data text file.

    Returns:
        user_datasets (dict of {user_id: [(sensor_values, activity_labels)]}):
            The processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}.
            The keys of the dictionary are the user_id (participant id).
            The values of the dictionary are lists of (sensor_values, activity_labels) pairs.
                - sensor_values are 2D numpy arrays of shape (length, channels=3).
                - activity_labels are 2D numpy arrays of shape (length, 2) with activity and source labels.
    """
    user_datasets = {}
    activity_map = {
        'Walking': 'walking',
        'Jogging': 'jogging',
        'Upstairs': 'walking_upstairs',
        'Downstairs': 'walking_downstairs',
        'Sitting': 'sitting',
        'Standing': 'standing'
    }

    all_data = []
    all_labels = []
    all_user_ids = []

    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, header=None, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'])

    # Ensure the columns 'x', 'y', and 'z' are floats
    df['x'] = df['x'].apply(lambda x: str(x).replace(';', '').strip()).astype(float)
    df['y'] = df['y'].apply(lambda x: str(x).replace(';', '').strip()).astype(float)
    df['z'] = df['z'].apply(lambda x: str(x).replace(';', '').strip()).astype(float)

    # Drop rows with NaN values
    df.dropna(subset=['x', 'y', 'z'], inplace=True)

    # Debugging: Print unique user IDs
    unique_user_ids = df['user'].unique()
    print(f"Unique user IDs in WISDM data: {unique_user_ids}")

    for user_id, user_data in df.groupby('user'):
        user_id = int(user_id)
        if user_id not in user_datasets:
            user_datasets[user_id] = []

        for activity, activity_data in user_data.groupby('activity'):
            if activity in activity_map:
                sensor_values = activity_data[['x', 'y', 'z']].values
                activity_labels = np.array([[activity_map[activity], 'phone_acc']] * len(sensor_values))

                all_data.append(sensor_values)
                all_labels.append(activity_labels)
                all_user_ids.append(np.repeat(user_id, len(sensor_values)))

    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    all_user_ids = np.concatenate(all_user_ids)

    # Debugging: Check the shapes of concatenated arrays
    print(f"All data shape: {all_data.shape}")
    print(f"All labels shape: {all_labels.shape}")
    print(f"All user IDs shape: {all_user_ids.shape}")

    normalized_data = data_pre_processing.normalise_data(all_data)

    for user_id in np.unique(all_user_ids):
        user_indices = np.where(all_user_ids == user_id)[0]
        user_data = normalized_data[user_indices]
        user_labels = all_labels[user_indices]
        if user_id not in user_datasets:
            user_datasets[user_id] = []
        user_datasets[user_id].append((user_data, user_labels))

    # Debugging: Print the processed user IDs
    print("WISDM user IDs after processing:", list(user_datasets.keys()))
    return user_datasets