import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys
import glob

IM_SHAPE = (64, 64, 3)


def plot_image_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(np.squeeze(img), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )


def plot_value_prediction(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


class TrainingDatasetLoader(object):
    def __init__(self, data_path, channels_last=True):
        print("Opening {}".format(data_path))
        sys.stdout.flush()

        self.cache = h5py.File(data_path, "r")

        print("Loading data into memory...")
        sys.stdout.flush()
        self.images = self.cache["images"][:]
        self.channels_last = channels_last
        self.labels = self.cache["labels"][:].astype(np.float32)
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]

        self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]

    def get_train_size(self):
        return self.train_inds.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size() // factor // batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            selected_inds = np.random.choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos
            )
        else:
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=n // 2, replace=False, p=p_pos
            )
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=n // 2, replace=False, p=p_neg
            )
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :, :, ::-1] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]

        if not self.channels_last:
            train_img = np.ascontiguousarray(
                np.transpose(train_img, (0, 3, 1, 2))
            )  # [B, H, W, C] -> [B, C, H, W]
        return (
            (train_img, train_label, sorted_inds)
            if return_inds
            else (train_img, train_label)
        )

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[: 10 * n : 10]]
        return (self.images[most_prob_inds, ...] / 255.0).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]


def get_test_faces(channels_last=True):
    cwd = os.path.dirname(__file__)
    images = {"LF": [], "LM": [], "DF": [], "DM": []}
    for key in images.keys():
        files = glob.glob(os.path.join(cwd, "data", "faces", key, "*.png"))
        for file in sorted(files):
            image = cv2.resize(cv2.imread(file), (64, 64))[:, :, ::-1] / 255.0
            if not channels_last:
                image = np.transpose(image, (2, 0, 1))
            images[key].append(image)

    return images["LF"], images["LM"], images["DF"], images["DM"]
