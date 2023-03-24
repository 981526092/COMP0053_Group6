from scipy.io import loadmat
import numpy as np
from keras.utils import to_categorical
from data_augmentation import augment_data
import os

# Function to get filenames without extension in the given directory
def get_filenames_without_extension(directory):
    filenames = os.listdir(directory)
    filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames]
    return filenames_without_extension

# Function to calculate the mode with a given threshold (default: 0.5) for a list of binary values (0 or 1)
def mode_threshold(list,threshold = 0.5):
    count = 0
    for i in list:
        if (i == 0):
            count += 1
    if count >= len(list)*threshold:
        return 0
    else:
        return 1

# Function to segment data using sliding window segmentation
def segment_data(data: np.ndarray, window_size: int = 180, overlap_ratio: float = 0.75, by_type: bool = True, min_frame: int = 12) -> np.ndarray:
    """
    Segments data using sliding window segmentation.
    Parameters:
        data (np.ndarray): Input data as a numpy array.
        window_size (int): Size of the sliding window.
        overlap_ratio (float): Overlap ratio of the sliding windows.
        by_type (bool): Flag to specify if the segmentation is by type.
        min_frame (int): Minimum frames required in a window.

    Returns:
        np.ndarray: A numpy array containing the segmented data.
    """
    # Input data validation
    assert data.shape[0] > 0
    assert window_size > 0
    assert 0 <= overlap_ratio < 1
    assert 0 <= min_frame < window_size
    dim = data.shape[1]
    instances = []
    # If not by type, append the data as a single instance
    if not by_type:
        instances.append(data)
    else:
        # Check if the data contains the required number of features
        assert data.shape[1] >= 71
        num_data = data.shape[0]
        left, right = 0, 1
        pre_type = -1
        cur_type = data[left, 70]
        # Iterate through the data and create instances for each type
        while right < num_data:
            if data[right, 70] == cur_type:
                right += 1
                continue
            if right - left <= min_frame:
                left = right
                cur_type = data[left, 70]
                right += 1
                continue
            new_instance = np.take(data, range(left, right), axis=0)
            if pre_type == new_instance[0, 70]:
                instances[-1] = np.vstack([instances[-1], new_instance])
            else:
                instances.append(new_instance)
            left = right
            pre_type = cur_type
            cur_type = data[left, 70]
            right += 1
        new_instance = np.take(data, range(left, right), axis=0)
        last = instances[-1]
        if last[0, 70] == new_instance[0, 70]:
            instances[-1] = np.vstack([last, new_instance])
        else:
            instances.append(new_instance)
    # Get the step size for sliding window segmentation
    step_size = int(window_size * (1 - overlap_ratio))
    windows = []
    # Iterate through instances and create windows using sliding window segmentation
    for instance in instances:
        if instance.shape[0] < window_size:
            instance = np.vstack([instance, np.zeros((window_size - instance.shape[0], dim))])
            windows.append(instance)
            continue
        if (instance.shape[0] - window_size) % step_size != 0:
            pad_size = step_size - (instance.shape[0] - window_size) % step_size
            instance = np.vstack([instance, np.zeros((pad_size, dim))])
        for i in range(0, instance.shape[0] - window_size + 1, step_size):
            windows.append(np.take(instance, range(i, i + window_size), axis=0))
    return np.array(windows)

# Function to load data and perform preprocessing
def load_data(filenames, data_set, downsampling = False, angle_energy = False,augment = False,data_path = '../CoordinateData/'):
    """
    Loads data from .mat files and performs preprocessing.
    Parameters:
        filenames (list): List of filenames to load.
        data_set (str): Name of the dataset.
        downsampling (bool): Flag to specify if downsampling should be applied.
        angle_energy (bool): Flag to specify if angle and energy features should be used.
        augment (bool): Flag to specify if data augmentation should be performed.
        data_path (str): Path to the dataset directory.
    Returns:
        tuple: A tuple containing the preprocessed X and y data.
    """
    # Initialize empty lists to hold training data and labels
    X_train_list = []
    y_train_list = []
    selected_data_list = []
    # Load and preprocess each file in filenames
    for file_name in filenames:
        # Load the file
        traindata = loadmat(data_path + data_set + '/' + file_name + '.mat')
        # Count the occurrences of each label in the file
        count_table = np.unique(traindata['data'][:,72:73],return_counts=True)
        # If downsampling is enabled, skip files with only one label
        if downsampling == True:
            if (len(count_table[0])) == 2:
                selected_data_list.append(traindata['data'])
                print(file_name + ' is selected to be used for training (downsampling)')
            else:
                print(file_name + ' is not selected to be used for training (downsampling)')
        else:
            selected_data_list.append(traindata['data'])
    # Segment each file into smaller windows and extract features and labels
    for i in range(len(selected_data_list)):
        # Segment the data into windows
        processed_data = segment_data(selected_data_list[i], window_size=180, overlap_ratio=0.75, by_type=True, min_frame=12)
        # Extract features and labels
        if angle_energy:
            X_segmented = np.concatenate((processed_data[:,:,78:104], processed_data[:,:,66:70]), axis=2)
        else:
            X_segmented = processed_data[:,:,0:70]
        y_segmented = processed_data[:,:,72:73]
        y_segmented = np.apply_along_axis(lambda x: mode_threshold(x), 1, y_segmented[:, :, 0])
        X_train_list.append(X_segmented)
        y_train_list.append(y_segmented)
    # Concatenate the training data and labels
    X_train_concat = np.concatenate(X_train_list, axis=0)
    y_train_concat = np.concatenate(y_train_list, axis=0)
    num_classes = 2  # classes
    # Convert the labels to one-hot encoding
    y_train_concat = to_categorical(y_train_concat, num_classes)
    # If data augmentation is enabled, augment the training data
    if augment == True:
        X_train_concat, y_train_concat = augment_data(X_train_concat, y_train_concat)
    return X_train_concat, y_train_concat

# Function to flatten input data (X) and convert one-hot encoded labels (y) back to integer class labels
def flatten_data(X,y):
    X_return = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    Y_return = np.argmax(y, axis=1)
    return X_return, Y_return

# Function to load raw data for specified participant numbers and data type
def load_raw_data(participant_num, data_type,data_path = '../CoordinateData/'):
    data = []
    for i in participant_num:
        data.append(loadmat(data_path+data_type+"/"+i+".mat")['data'])
    return np.concatenate(data, axis=0)

