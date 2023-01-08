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


class DatasetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, training=True):

        print("Opening {}".format(data_path))
        sys.stdout.flush()

        self.cache = h5py.File(data_path, "r")

        print("Loading data into memory...")
        sys.stdout.flush()
        self.images = self.cache["images"][:]
        self.labels = self.cache["labels"][:].astype(np.float32)
        self.image_dims = self.images.shape

        train_inds = np.arange(len(self.images))
        pos_train_inds = train_inds[self.labels[train_inds, 0] == 1.0]
        neg_train_inds = train_inds[self.labels[train_inds, 0] != 1.0]
        if training:
            self.pos_train_inds = pos_train_inds[: int(0.8 * len(pos_train_inds))]
            self.neg_train_inds = neg_train_inds[: int(0.8 * len(neg_train_inds))]
        else:
            self.pos_train_inds = pos_train_inds[-1 * int(0.2 * len(pos_train_inds)) :]
            self.neg_train_inds = neg_train_inds[-1 * int(0.2 * len(neg_train_inds)) :]

        np.random.shuffle(self.pos_train_inds)
        np.random.shuffle(self.neg_train_inds)

        self.train_inds = np.concatenate((self.pos_train_inds, self.neg_train_inds))
        self.batch_size = batch_size
        self.p_pos = np.ones(self.pos_train_inds.shape) / len(self.pos_train_inds)

    def get_train_size(self):
        return self.pos_train_inds.shape[0] + self.neg_train_inds.shape[0]

    def __len__(self):
        return int(np.floor(self.get_train_size() / self.batch_size))

    def __getitem__(self, index):
        selected_pos_inds = np.random.choice(
            self.pos_train_inds, size=self.batch_size // 2, replace=False, p=self.p_pos
        )
        selected_neg_inds = np.random.choice(
            self.neg_train_inds, size=self.batch_size // 2, replace=False
        )
        selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]
        return np.array(train_img), np.array(train_label)

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[: 10 * n : 10]]
        return (self.images[most_prob_inds, ...] / 255.0).astype(np.float32)

    def get_all_faces(self):
        return (self.images[self.pos_train_inds] / 255.0).astype(np.float32)

    def return_sample_batch(self):
        return self.__getitem__(0)


def get_test_faces():
    cwd = os.path.dirname(__file__)
    images = {"LF": [], "LM": [], "DF": [], "DM": []}
    for key in images.keys():
        files = glob.glob(os.path.join(cwd, "data", "faces", key, "*.png"))
        for file in sorted(files):
            image = cv2.resize(cv2.imread(file), (64, 64))[:, :, ::-1] / 255.0
            images[key].append(image)

    return images["LF"], images["LM"], images["DF"], images["DM"]


def plot_k(imgs, fname=None):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6)
    num_images = len(imgs)
    for img in range(num_images):
        ax = fig.add_subplot(int(num_images / 5), 5, img + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img_to_show = imgs[img]
        ax.imshow(img_to_show, interpolation="nearest")
    plt.subplots_adjust(wspace=0.20, hspace=0.20)
    plt.show()
    if fname:
        plt.savefig(fname)
    plt.clf()


def plot_percentile(imgs, fname=None):
    fig = plt.figure()
    fig, axs = plt.subplots(1, len(imgs), figsize=(11, 8))
    for img in range(len(imgs)):
        ax = axs[img]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img_to_show = imgs[img]
        ax.imshow(img_to_show, interpolation="nearest")
    if fname:
        plt.savefig(fname)


def plot_accuracy_vs_risk(sorted_images, sorted_uncertainty, sorted_preds, plot_title):
    num_percentile_intervals = 10
    num_samples = len(sorted_images) // num_percentile_intervals
    all_imgs = []
    all_unc = []
    all_acc = []
    for percentile in range(num_percentile_intervals):
        cur_imgs = sorted_images[
            percentile * num_samples : (percentile + 1) * num_samples
        ]
        cur_unc = sorted_uncertainty[
            percentile * num_samples : (percentile + 1) * num_samples
        ]
        cur_predictions = tf.nn.sigmoid(
            sorted_preds[percentile * num_samples : (percentile + 1) * num_samples]
        )
        avged_imgs = tf.reduce_mean(cur_imgs, axis=0)
        all_imgs.append(avged_imgs)
        all_unc.append(tf.reduce_mean(cur_unc))
        all_acc.append((np.ones((num_samples)) == np.rint(cur_predictions)).mean())

    plt.plot(np.arange(num_percentile_intervals) * 10, all_acc)
    plt.title(plot_title)
    plt.show()
    plt.clf()
    return all_imgs
