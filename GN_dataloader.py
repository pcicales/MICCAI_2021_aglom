import os
import random
import torch
from torchvision import transforms
from config import options
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(options.gpu_used)
device = torch.device("cuda" if options.cuda else "cpu")

class GN_Dataset(Dataset):
    def __init__(self, mode="train", input_size=(options.img_w, options.img_h), test=0):
        # self.imgs_dir = options.dataset  # Path to AMR dataset
        if options.datamode:
            self.cologne_imgs_dir = options.dataset_dir + '/HULA/Cologne_GN/multi/'
            self.szeged_imgs_dir = options.dataset_dir + '/HULA/Szeged_GN/multi/'
        else:
            self.cologne_imgs_dir = options.dataset_dir + '/HULA/Cologne_GN/img/'
            self.szeged_imgs_dir = options.dataset_dir + '/HULA/Szeged_GN/img/'
        self.mode = mode
        self.input_size = input_size
        self.label_code = {"PGNMID": 0, "Fibrillary": 1, "ANCA": 2, "Membranous": 3, "IgAGN": 4,
                           'ABMGN': 5, 'SLEGN-IV': 6, 'IAGN': 7, 'DDD': 8, 'MPGN': 9}
        self.patients = {}
        self.patient_labels = {}
        self.folds = options.dataset_dir + '/HULA/folds/GN_all/'
        self.test = test

        self.glom_files = []
        self.glom_ids = []
        self.glom_labels = []

        if self.mode == "train":
            set = list(range(5))
            set.remove(options.test_fold_val)
            for i in set:
                self.glom_files = self.glom_files + np.load(self.folds + 'GN_FOLD_{}.npz'.format(i), allow_pickle=True)['FILES'].tolist()
                self.glom_ids = self.glom_ids + np.load(self.folds + 'GN_FOLD_{}.npz'.format(i), allow_pickle=True)['ID'].tolist()
                self.glom_labels = self.glom_labels + np.load(self.folds + 'GN_FOLD_{}.npz'.format(i), allow_pickle=True)['CL'].tolist()
            print('Training on 4 folds, image count:', len(self.glom_files))
        elif self.mode == "val":
            self.glom_files = np.load(self.folds + 'GN_FOLD_{}.npz'.format(options.test_fold_val), allow_pickle=True)['FILES'].tolist()
            self.glom_ids = np.load(self.folds + 'GN_FOLD_{}.npz'.format(options.test_fold_val), allow_pickle=True)['ID'].tolist()
            self.glom_labels = np.load(self.folds + 'GN_FOLD_{}.npz'.format(options.test_fold_val), allow_pickle=True)['CL'].tolist()
            print("Testing image count:", len(self.glom_files))
        elif self.mode == "test":
            self.glom_files = np.load(self.folds + 'GN_FOLD_{}.npz'.format(self.test), allow_pickle=True)['FILES'].tolist()
            self.glom_ids = np.load(self.folds + 'GN_FOLD_{}.npz'.format(self.test), allow_pickle=True)['ID'].tolist()
            self.glom_labels = np.load(self.folds + 'GN_FOLD_{}.npz'.format(self.test), allow_pickle=True)['CL'].tolist()
            print("Testing image count:", len(self.glom_files))

        # create label dict
        for key, val in zip(self.glom_ids, self.glom_labels):
            self.patient_labels.setdefault(key, []).append(val)

        # Create patient dictionary
        for key, val in zip(self.glom_ids, self.glom_files):
            self.patients.setdefault(key, []).append(val)

    def __len__(self):
        return len(self.patients)  # Length of dataset is number of patients

    def __getitem__(self, index):
        full_files_cologne = [f for f in os.listdir(self.cologne_imgs_dir[:-1]) if os.path.isfile(os.path.join(self.cologne_imgs_dir[:-1], f))]
        full_files_szeged = [f for f in os.listdir(self.szeged_imgs_dir[:-1]) if os.path.isfile(os.path.join(self.szeged_imgs_dir[:-1], f))]
        full_files = full_files_cologne + full_files_szeged
        # getitem() builds one stack for a single patient, patient indexed by given index
        existing_patient_images = self.patients[list(self.patients.keys())[index]][0]
        patient_samples = []
        # Sample options.stack_size images from this patient
        target_label = self.patient_labels[list(self.patients.keys())[index]][0]
        if len(existing_patient_images) < options.stack_size:
            # Take all existing images, shuffle to randomize order
            random.shuffle(existing_patient_images)
            for img_name in existing_patient_images:
                matching = [s for s in full_files if img_name[:-4] in s]
                try:
                    img = Image.open(self.cologne_imgs_dir + matching[0])
                except:
                    img = Image.open(self.szeged_imgs_dir + matching[0])

                img = transforms.ToTensor()(img)

                if self.mode == 'train':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
                    img = transforms.RandomRotation(90, fill=(0,))(img)  # resample=PIL.Image.BICUBIC

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                patient_samples.append(img)

            for i in range(options.stack_size - len(existing_patient_images)):
                # Sampling with replacement, ToDo: Add in option for sampling without replacement
                sampled_img = random.sample(existing_patient_images, 1)
                # existing_files.remove(original[0])
                file_name = sampled_img[0]
                matching = [s for s in full_files if file_name[:-4] in s]
                try:
                    img = Image.open(self.cologne_imgs_dir + matching[0])
                except:
                    img = Image.open(self.szeged_imgs_dir + matching[0])

                img = transforms.ToTensor()(img)

                if self.mode == 'train':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
                    img = transforms.RandomRotation(90, fill=(0,))(img)  # resample=PIL.Image.BICUBIC

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                patient_samples.append(img)
        else:

            # Patient has more images than options.stack_size, sample stack_size images from the vector
            for _ in range(options.stack_size):
                # Sampling without replacement
                original = random.sample(existing_patient_images, 1)
                existing_patient_images.remove(original[0])
                file_name = original[0]

                matching = [s for s in full_files if file_name[:-4] in s]
                try:
                    img = Image.open(self.cologne_imgs_dir + matching[0])
                except:
                    img = Image.open(self.szeged_imgs_dir + matching[0])

                img = transforms.ToTensor()(img)

                if self.mode == 'train':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
                    img = transforms.RandomRotation(90, fill=(0,))(img)  # resample=PIL.Image.BICUBIC

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                patient_samples.append(img)

        return torch.stack(patient_samples, dim=0), self.label_code[target_label]
