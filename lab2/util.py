import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys

IM_SHAPE = (64, 64, 3)


class TrainingDatasetLoader(object):
    def __init__(self, data_path):

        print "Opening {}".format(data_path)
        sys.stdout.flush()

        self.cache = h5py.File(data_path, 'r')

        print "Loading data into memory..."
        sys.stdout.flush()
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:]
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]

        self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = self.train_inds[ self.labels[self.train_inds, 0] == 1.0 ]
        self.neg_train_inds = self.train_inds[ self.labels[self.train_inds, 0] != 1.0 ]

    def get_train_size(self):
        return self.train_inds.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size()//factor//batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            selected_inds = np.random.choice(self.pos_train_inds, size=n, replace=False, p=p_pos)
        else:
            selected_pos_inds = np.random.choice(self.pos_train_inds, size=n//2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(self.neg_train_inds, size=n//2, replace=False, p=p_neg)
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = self.images[sorted_inds,:,:,::-1]/255.
        train_label = self.labels[sorted_inds,...]
        return (train_img, train_label, sorted_inds) if return_inds else (train_img, train_label)

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
        return self.images[most_prob_inds,...]/255.

    def get_all_train_faces(self):
        return self.images[ self.pos_train_inds ]



class PPBFaceEvaluator:
    ''' Evaluate on the PPB dataset'''
    def __init__(self, skip=4):

        path_to_faces = tf.keras.utils.get_file('ppb', 'https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1', extract=True)
        self.ppb_root = os.path.join(os.path.split(path_to_faces)[0], 'PPB-2017')

        ppb_anno = os.path.join(self.ppb_root,'PPB-2017-metadata.csv')

        self.anno_dict = {}
        with open(ppb_anno) as f:
            for line in f.read().split('\r'):
                ind, name, gender, numeric, skin, country = line.split(',')
                self.anno_dict[name] = (gender.lower(),skin.lower())

        image_dir = os.path.join(self.ppb_root, "imgs")
        image_files = sorted(os.listdir(image_dir))[::skip] #sample every 4 images for computation time in the lab

        self.raw_images = {
            'male_darker':[],
            'male_lighter':[],
            'female_darker':[],
            'female_lighter':[],
        }

        for filename in image_files:
            if not filename.endswith(".jpg"):
                continue
            image = cv2.imread(os.path.join(image_dir,filename))[:,:,::-1]
            gender, skin = self.anno_dict[filename]
            self.raw_images[gender+'_'+skin].append(image)


    def get_sample_faces_from_demographic(self, gender, skin_color):
        key = self.__get_key(gender, skin_color)
        data = self.raw_images[key][50]/255.
        return data


    def evaluate(self, models_to_test, gender, skin_color, output_idx=None, from_logit=False, patch_stride=0.2, patch_depth=5):
        correct_predictions = [0.0]*len(models_to_test)

        key = self.__get_key(gender, skin_color)
        num_faces = len(self.raw_images[key])

        import progressbar
        bar = progressbar.ProgressBar()
        for face_idx in bar(range(num_faces)):

            image = self.raw_images[key][face_idx]
            height, width, _ = image.shape

            patches, bboxes = slide_square(image, patch_stride, width/2, width, patch_depth)
            patches = tf.cast(tf.constant(patches, dtype=tf.uint8), tf.float32)/255.

            for model_idx, model in enumerate(models_to_test):
                out = model(patches)
                y = out if output_idx is None else out[output_idx]
                y = y.numpy()
                y_inds = np.argsort(y.flatten())
                most_likely_prob = y[y_inds[-1]]
                if (from_logit and most_likely_prob >= 0.0) or \
                   (not from_logit and most_likely_prob >= 0.5):
                        correct_predictions[model_idx] += 1

        accuracy = [correct_predictions[i]/num_faces for i,_ in enumerate(models_to_test)]
        return accuracy


    def __get_key(self, gender, skin_color):
        gender = gender.lower()
        skin_color = skin_color.lower()
        assert gender in ['male', 'female']
        assert skin_color in ['lighter', 'darker']
        return '{}_{}'.format(gender, skin_color)



''' function to slide a square across image and extract square regions
img = the image
stride = (0,1], provides the fraction of the dimension for which to slide to generate a crop
max_size = maximum square size
min_size = minimum square size
n = number of different sizes including min_size, max_size '''
def slide_square(img, stride, min_size, max_size, n):
    img_h, img_w = img.shape[:2]

    square_sizes = np.linspace(min_size, max_size, n, dtype=np.int32)
    square_images = []
    square_bbox = [] # list of list of tuples: [(i1,j1), (i2,j2)] where i1,j1 is the top left corner; i2,j2 is bottom right corner
    # for each of the square_sizes
    for level, sq_dim in enumerate(square_sizes):

        stride_length = int(stride*sq_dim)
        stride_start_i = xrange(0,int(img_h-sq_dim+1),stride_length)
        stride_start_j = xrange(0,int(img_w-sq_dim+1),stride_length)
        for i in stride_start_i:
            for j in stride_start_j:
                square_top_left = (i,j)

                square_bottom_right = (i+sq_dim,j+sq_dim)

                square_corners = (square_top_left,square_bottom_right)
                square_corners_show = ((j,i),(j+sq_dim,i+sq_dim))
                square_image = img[i:i+sq_dim,j:j+sq_dim]

                square_resize = cv2.resize(square_image, IM_SHAPE[:2], interpolation=cv2.INTER_NEAREST)
                # append to list of images and bounding boxes

                square_images.append(square_resize)
                square_bbox.append(square_corners)

    return square_images, square_bbox
