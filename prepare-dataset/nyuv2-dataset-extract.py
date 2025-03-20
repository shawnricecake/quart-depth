from __future__ import print_function

import h5py
import numpy as np
import os
import scipy.io
import sys
import cv2
# from pyxform.utils import unichr

def convert_image(i, scene, depth_raw, image, depth_dense):

    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    dense_folder = folder + '/dense'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(dense_folder):
        os.makedirs(dense_folder)

    img_depth = depth_raw * 1000.0
    img_depth_uint16 = img_depth.astype(np.uint16)
    dense_depth = depth_dense * 1000.0
    dense_depth_uint16 = dense_depth.astype(np.uint16)

    cv2.imwrite("%s/sync_depth_%05d.png" % (folder, i), img_depth_uint16)
    cv2.imwrite("%s/sync_depth_dense_%05d.png" % (dense_folder, i), dense_depth_uint16)
    image = image[:, :, ::-1]
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), image_black_boundary)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    depth_raw = h5_file['rawDepths']
    depth_dense = h5_file['depths']

    print("reading", sys.argv[1])

    images = h5_file['images']
    # scenes = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]
    scenes = [''.join(chr(c) if isinstance(c, int) else chr(int(c)) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    print("processing images")
    for i, image in enumerate(images):
        print("image", i + 1, "/", len(images))
        convert_image(i, scenes[i], depth_raw[i, :, :].T, image.T, depth_dense[i,:,:].T)

    print("Finished")


