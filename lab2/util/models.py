import tensorflow as tf

class PneumothoraxDetectionModel:
    def __init__(self):

        self.x = tf.placeholder(shape=[None, 256, 256, 1], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 2], dtype=tf.float32)

        self.x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x)

        self.conv1 = tf.layers.conv2d(inputs=self.x_image, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.skip1 = tf.layers.max_pooling2d(self.conv1+self.conv2, 2, 2)

        self.conv3 = tf.layers.conv2d(inputs=self.skip1, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.conv4 = tf.layers.conv2d(inputs=self.conv3, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.skip2 = tf.layers.max_pooling2d(self.conv3+self.conv4, 2, 2)

        self.conv5 = tf.layers.conv2d(inputs=self.skip2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv6 = tf.layers.conv2d(inputs=self.conv5, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.skip3 = tf.layers.max_pooling2d(self.conv5+self.conv6, 2, 2)

        self.conv7 = tf.layers.conv2d(inputs=self.skip3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv8 = tf.layers.conv2d(inputs=self.conv7, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.skip4 = tf.layers.max_pooling2d(self.conv7+self.conv8, 2, 2)

        self.g_avg_pool = tf.reduce_mean(self.skip4, [1,2])

        self.y_ = tf.layers.dense(self.g_avg_pool, 2)
        self.probabilities = tf.nn.softmax(self.y_)

