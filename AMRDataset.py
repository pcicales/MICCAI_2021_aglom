import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from config import options
from torch.utils.data import Dataset
from collections import Counter
import random
from utils.logger_utils import Logger
import PIL

import warnings
warnings.filterwarnings("ignore")


def log_string(log_fout, out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)


def remove_nested(L, val):
    for i in range(len(L)):
        L[i] = list(filter((val).__ne__, L[i]))
    return L


def find_tcons(L): # Generating the consensus labels for the label sampling task, test/val set only
    for i in range(len(L)):
        L[i] = most_frequent(L[i])
    return L


def most_frequent(L):  # Can modify the order here depending on whether we prefer fewer False positives
    # or False negatives, may also consider doing this based on the labelers.
    if Counter(L).most_common(1)[0][1] <= len(L)//2 and \
            ('AMR' in [item for t in Counter(L).most_common(2) for item in t]):
        return 'AMR'
    elif Counter(L).most_common(1)[0][1] <= len(L)//2 and \
            ('Non-AMR' in [item for t in Counter(L).most_common(2) for item in t]):
        return 'Non-AMR'
    elif Counter(L).most_common(1)[0][1] <= len(L)//2 and \
            ('Inconclusive' in [item for t in Counter(L).most_common(2) for item in t]):
        return 'Inconclusive'
    else:
        return Counter(L).most_common(1)[0][0]


class AMRDataset(Dataset):
    def __init__(self, mode="train", input_size=(256, 256), LOG_FOUT=None):  # Could have data_len parameter
        self.imgs_dir = options.dataset  # Path to AMR dataset
        self.mode = mode
        self.input_size = input_size
        self.label_mode = options.label_mode
        self.val = options.test_fold_val
        self.label_code = {"Non-AMR": 0, "AMR": 1, "Inconclusive": 2, "None": 3}
        self.log_fout = LOG_FOUT

        t_labels = []
        v_labels = []
        t_images = []
        v_images = []

        for i in range(5):
            if options.num_classes == 3:
                if options.label_mode == 'cons':
                    if i == options.test_fold_val:
                        v_images = v_images + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
                        v_labels = v_labels + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['CONS'].tolist()

                    else:
                        t_images = t_images + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
                        t_labels = t_labels + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['CONS'].tolist()

                elif options.label_mode == 'samp':
                    if i == options.test_fold_val:
                        v_images = v_images + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
                        v1_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L1'].tolist()
                        v1_labels = [list(options.native_labels * a) for a in zip(v1_labels)]
                        v2_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L2'].tolist()
                        v3_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L3'].tolist()
                        v4_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L4'].tolist()
                        v_labels_raw = [a + [b] + [c] + [d] for a, b, c, d in
                                        zip(v1_labels, v2_labels, v3_labels, v4_labels)]
                        v_labels_raw = remove_nested(v_labels_raw, "None")
                        v_labels_raw = find_tcons(v_labels_raw)
                        v_labels = v_labels + v_labels_raw

                    else:
                        t_images = t_images + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
                        t1_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L1'].tolist()
                        t1_labels = [list(options.native_labels * a) for a in zip(t1_labels)]
                        t2_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L2'].tolist()
                        t3_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L3'].tolist()
                        t4_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L4'].tolist()
                        t_labels_raw = [a + [b] + [c] + [d] for a, b, c, d in
                                        zip(t1_labels, t2_labels, t3_labels, t4_labels)]
                        t_labels_raw = remove_nested(t_labels_raw, "None")
                        t_labels = t_labels + t_labels_raw

                else:
                    raise ValueError('AMR can only follow consensus (cons) or label sampling (samp) schemes.')

            elif options.num_classes == 2:
                if options.label_mode == 'cons':
                    if i == options.test_fold_val:
                        v_images_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES']
                        v_labels_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['CONS']
                        v_images_raw = list(v_images_raw[v_labels_raw != 'Inconclusive'])
                        v_labels_raw = list(v_labels_raw[v_labels_raw != 'Inconclusive'])
                        v_images = v_images + v_images_raw
                        v_labels = v_labels + v_labels_raw

                    else:
                        t_images_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES']
                        t_labels_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['CONS']
                        t_images_raw = list(t_images_raw[t_labels_raw != 'Inconclusive'])
                        t_labels_raw = list(t_labels_raw[t_labels_raw != 'Inconclusive'])
                        t_images = t_images + t_images_raw
                        t_labels = t_labels + t_labels_raw

                elif options.label_mode == 'samp':
                    if i == options.test_fold_val:
                        v_images_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES']
                        v1_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L1'].tolist()
                        v1_labels = [list(options.native_labels * a) for a in zip(v1_labels)]
                        v2_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L2'].tolist()
                        v3_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L3'].tolist()
                        v4_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L4'].tolist()
                        v_labels_raw = [a + [b] + [c] + [d] for a, b, c, d in
                                        zip(v1_labels, v2_labels, v3_labels, v4_labels)]
                        v_labels_raw = remove_nested(v_labels_raw, "None")
                        v_labels_cons = np.array(find_tcons(v_labels_raw.copy()))
                        v_images_raw = list(np.array(v_images_raw)[v_labels_cons != 'Inconclusive'])
                        v_labels_raw = list(np.array(v_labels_cons)[v_labels_cons != 'Inconclusive'])
                        v_labels = v_labels + v_labels_raw
                        v_images = v_images + v_images_raw

                    else:
                        t_images_raw = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES']
                        t1_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L1'].tolist()
                        t1_labels = [list(options.native_labels * a) for a in zip(t1_labels)]
                        t2_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L2'].tolist()
                        t3_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L3'].tolist()
                        t4_labels = np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['L4'].tolist()
                        t_labels_raw = [a + [b] + [c] + [d] for a, b, c, d in
                                        zip(t1_labels, t2_labels, t3_labels, t4_labels)]
                        t_labels_raw = remove_nested(t_labels_raw, "None")
                        t_labels_cons = np.array(find_tcons(t_labels_raw.copy()))
                        t_images_raw = list(np.array(t_images_raw)[t_labels_cons != 'Inconclusive'])
                        t_labels_raw = list(np.array(t_labels_raw)[t_labels_cons != 'Inconclusive'])
                        t_labels_raw = remove_nested(t_labels_raw, "Inconclusive")
                        t_labels = t_labels + t_labels_raw
                        t_images = t_images + t_images_raw

                else:
                    raise ValueError('AMR can only follow consensus (cons) or label sampling (samp) schemes.')

            else:
                raise ValueError('AMR can only be a 2 or 3 class problem.')

        # Grab list of filenames in supplied dataset directory directory
        log_string(self.log_fout, str("Loading " + str(self.mode) + " data..."))

        # Labels and images lists now contain all the labels and image filenames for
        # images in the dataset directory
        # log_string(self.log_fout, string info
        if self.mode == "train":
            self.labels = np.array(t_labels)
            self.images = np.array(t_images)
            log_string(self.log_fout, str("Training dataset images count: " + str(len(self.images))))
            log_string(self.log_fout, str("Training dataset labels count: " + str(len(self.labels))))
            if options.label_mode == 'samp':
                samp_count = np.array(find_tcons(t_labels.copy()))
                log_string(self.log_fout, str("Number of Non-AMR labels: " + str(len(np.where(samp_count == "Non-AMR")[0]))))
                log_string(self.log_fout, str("Number of AMR labels: " + str(len(np.where(samp_count == "AMR")[0]))))
                log_string(self.log_fout, str("Number of Inconclusive labels: " + str(len(np.where(samp_count == "Inconclusive")[0])) + '\n'))
            else:
                log_string(self.log_fout, str("Number of Non-AMR labels: " + str(len(np.where(self.labels == "Non-AMR")[0]))))
                log_string(self.log_fout, str("Number of AMR labels: " + str(len(np.where(self.labels == "AMR")[0]))))
                log_string(self.log_fout, str("Number of Inconclusive labels: " + str(len(np.where(self.labels == "Inconclusive")[0])) + '\n'))

        else:
            self.labels = np.array(v_labels)
            self.images = np.array(v_images)
            log_string(self.log_fout, "Testing dataset images count: " + str(len(self.images)))
            log_string(self.log_fout, "Testing dataset labels count: " + str(len(self.labels)))
            log_string(self.log_fout, "Number of Non-AMR labels: " + str(len(np.where(self.labels == "Non-AMR")[0])))
            log_string(self.log_fout, "Number of AMR labels: " + str(len(np.where(self.labels == "AMR")[0])))
            log_string(self.log_fout, "Number of Inconclusive labels: " + str(len(np.where(self.labels == "Inconclusive")[0])) + '\n')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Open image
        image_name = self.images[index]
        img = Image.open(self.imgs_dir + image_name)  # Are other conversions needed?
        target_label = self.labels[index]  # label_bunch for the image indexed
        if self.mode == 'train' and options.label_mode == 'samp':
            target_label = random.choice(target_label)

        # Min-max scaling
        img = transforms.ToTensor()(img)
        # img = img.div(255.)
        img = (img - img.min()) / (img.max() - img.min())

        if self.mode == 'train':
            # normalization & augmentation
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)  # Bilinear resizing?
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.RandomResizedCrop(self.input_size[0], scale=(0.7, 1.))(img)
            # img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            # img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)

        if self.mode == 'val':
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        if self.mode == 'test':
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, self.label_code[target_label]
