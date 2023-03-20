from scipy.io import loadmat
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle

def jitter(X, y, std_dev):
    noise = np.random.normal(0, std_dev, X.shape)
    return X + noise, y

def crop_new(X, y, probability):
    mask = np.random.rand(*X.shape) < probability
    return X * (1 - mask), y

def augment_data(X, y):
    X_jittered_1, y_jittered_1 = jitter(X, y, 0.05)
    X_jittered_2, y_jittered_2 = jitter(X, y, 0.1)

    X_cropped_1, y_cropped_1 = crop_new(X, y, 0.05)
    X_cropped_2, y_cropped_2 = crop_new(X, y, 0.1)

    X_augmented = np.concatenate((X, X_jittered_1, X_jittered_2, X_cropped_1, X_cropped_2), axis=0)
    y_augmented = np.concatenate((y, y_jittered_1, y_jittered_2, y_cropped_1, y_cropped_2), axis=0)

    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=0)

    return X_augmented, y_augmented

def mode_threshold(list,threshold = 0.5):
    count = 0
    for i in list:
        if (i == 0):
            count += 1
    if count >= len(list)*threshold:
        return 0
    else:
        return 1

def new_segment_data(data: np.ndarray, window_size: int = 180, overlap_ratio: float = 0.75, by_type: bool = True, min_frame: int = 12) -> np.ndarray:
    assert data.shape[0] > 0
    assert window_size > 0
    assert 0 <= overlap_ratio < 1
    assert 0 <= min_frame < window_size

    dim = data.shape[1]
    instances = []

    if not by_type:
        instances.append(data)
    else:
        assert data.shape[1] >= 71

        num_data = data.shape[0]
        left, right = 0, 1
        pre_type = -1
        cur_type = data[left, 70]
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

    step_size = int(window_size * (1 - overlap_ratio))
    windows = []

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

def load_data(filenames, data_set, downsampling = False, angle_energy = False,augment = False):
    data_path = '../CoordinateData/'
    X_train_list = []
    y_train_list = []

    selected_data_list = []
    for file_name in filenames:
        traindata = loadmat(data_path + data_set + '/' + file_name + '.mat')
        count_table = np.unique(traindata['data'][:,72:73],return_counts=True)
        if downsampling == True:
            if (len(count_table[0])) == 2:
                selected_data_list.append(traindata['data'])
        else:
            selected_data_list.append(traindata['data'])

    for i in range(len(selected_data_list)):
        processed_data = new_segment_data(selected_data_list[i], window_size=180, overlap_ratio=0.75, by_type=True, min_frame=12)

        if angle_energy:
            X_segmented = np.concatenate((processed_data[:,:,78:104], processed_data[:,:,66:70]), axis=2)
        else:
            X_segmented = processed_data[:,:,0:70]
        y_segmented = processed_data[:,:,72:73]

        y_segmented = np.apply_along_axis(lambda x: mode_threshold(x), 1, y_segmented[:, :, 0])

        X_train_list.append(X_segmented)
        y_train_list.append(y_segmented)

    X_train_concat = np.concatenate(X_train_list, axis=0)
    y_train_concat = np.concatenate(y_train_list, axis=0)

    num_classes = 2  # classes

    y_train_concat = to_categorical(y_train_concat, num_classes)

    if augment == True:
        X_train_concat, y_train_concat = augment_data(X_train_concat, y_train_concat)

    return X_train_concat, y_train_concat
