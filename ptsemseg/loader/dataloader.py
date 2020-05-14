
import os
import torch
import sys
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.augmentations import *

class data_loader(data.Dataset):
    # 19classes, RGB of maskes

    label_colours = dict(zip(range(7), colors))

    # mean_rgb = {
    #     "pascal": [103.939, 116.779, 123.68],
    #     "cityscapes": [0.0, 0.0, 0.0],
    # }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
            self,
            root="/media/disk2/sombit/kitti_seg",
            # root="/home/sombit/kitti",
            split="train",
            is_transform=True,
            img_size=(375, 1242),
            augmentations=None,
            img_norm=True,
            saliency_eval_depth = False
            # version="cityscapes",
    ):
        """__init__
                :param root:
                :param split:
                :param is_transform:
                :param img_size:
                :param augmentations
                """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 13
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        # self.mean = np.array(self.mean_rgb[version])
        self.files = {}
        self.saliency_eval_depth = saliency_eval_depth    # for later saliency evaluation on depth, always set to False for KITTI segmentation

        if self.split == "test":
            self.images_base = os.path.join(self.root, "testing", "image_2")
            self.annotations_base = os.path.join(self.root, "training", "semantic")  # invalid
        else:
            self.images_base = os.path.join(self.root, "training", "image_2")
            self.annotations_base = os.path.join(self.root, "training", "semantic")

        self.all_files = os.listdir(self.images_base)
        self.all_files.sort()

        # split 40 images from the training set as the val set
        if self.split == "val":
            self.files[split] = self.all_files[::5]  # select one img from every 5 imgs into the val set
        # 160 training images
        if self.split == "train":
            # self.files[split] = [file_name for file_name in self.all_files if file_name not in self.all_files[::5]]
            self.files[split] = self.all_files
        
        if self.split == "test":
            self.files[split] = self.all_files


       self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1,7,11,17,19,20,21,31]
        self.valid_classes = [
            7,
            # 8,
            11,
            12,
            # 13,
            # 17,
            # 19,
            # 20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            # 31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            # "sidewalk",
            "building",
            "wall",
            # "fence",
            # "pole",
            # "traffic_light",
            # "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            # "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(13)))
        self.decode_class_map = dict(zip(range(13), self.valid_classes))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))
        sys.stdout.flush()

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        path = self.files[self.split][index].rstrip()
        img_path = os.path.join(self.images_base, path)
        lbl_path = os.path.join(self.annotations_base, path)

        img = m.imread(img_path)  # original image size: 375*1242*3
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)  # original label size: 375*1242
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, img_path

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        # img: shape: [h, w, 3]
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        if self.saliency_eval_depth == False:
            img = img[:, :, ::-1]  # RGB -> BGR  shape: [h, w, 3]
        img = img.astype(np.float64)
        # img -= self.mean
        if self.img_norm:
            if self.saliency_eval_depth == False:
                img = img.astype(float) / 255.0
            else:
                img = ((img / 255 - 0.5) / 0.5)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)  # shape: [3, h, w]

        classes = np.unique(lbl)  # all classes included in this label image
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
            # sys.stdout.flush()

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()  # tensor, shape: [3, h, w]
        lbl = torch.from_numpy(lbl).long()  # tensor, shape: [h, w]

        return img, lbl

    def decode_segmap_tocolor(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def decode_segmap_tolabelId(self, temp):
        labels_ID = temp.copy()
        for i in range(13):
            labels_ID[temp == i] = self.valid_classes[i]
        return labels_ID

    def encode_segmap(self, mask):
        # Put all void classes to 250
        # map valid classes to 0~18
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask