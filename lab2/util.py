import cv2
import os
import numpy as np

IM_SHAPE = (64, 64, 3)

# function to slide a square across image and extract square regions
# img = the image
# stride = (0,1], provides the fraction of the dimension for which to slide to generate a crop
# max_size = maximum square size
# min_size = minimum square size
# n = number of different sizes including min_size, max_size
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
