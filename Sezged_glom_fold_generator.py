import glob
import random
import numpy as np
import pandas as pd

GLOM_PATH = '/home/cougarnet.uh.edu/pcicales/Documents/data/HULA/Szeged_GN/multi/'
ALL_IMG = glob.glob(GLOM_PATH + '*.jpg')
SLIDE_ID = [''] * len(ALL_IMG)
FOLDS = 5

for i in range(len(ALL_IMG)):
    if '.scn' in ALL_IMG[i]:
        SLIDE_ID[i] = ALL_IMG[i].split('/')[-1].split('.scn')[0]
    elif '.ndpi' in ALL_IMG[i]:
        SLIDE_ID[i] = ALL_IMG[i].split('/')[-1].split('.ndpi')[0]
    else:
        name_list = ALL_IMG[i].split('/')[-1].split('-')
        if '-.jpg' in ALL_IMG[i]:
            SLIDE_ID[i] = '-'.join(name_list[0:-2])
        else:
            SLIDE_ID[i] = '-'.join(name_list[0:-1])

ID_LIST = list(set(SLIDE_ID))
random.shuffle(ID_LIST)
MAX_COUNT = 0
PAT_COUNT = 0
labels = pd.read_csv('/home/cougarnet.uh.edu/srizvi7/Desktop/AMR_project_style_transfer/AMR_main_summary.csv')

for i in range(FOLDS):
    TRAIN_FILES = []
    for id in ID_LIST[int(i * len(ID_LIST)//FOLDS):int((i+1) * len(ID_LIST)//FOLDS)]:
        if 'PAS' not in id:
            for s in ALL_IMG:
                if (id in s) and ('PAS' not in s):
                    sample = labels[labels['Glomerulus ID'] == s.split('/')[-1]].values.tolist()[0]
                    TRAIN_FILES.append(sample)
        else:
            for s in ALL_IMG:
                if (id in s):
                    sample = labels[labels['Glomerulus ID'] == s.split('/')[-1]].values.tolist()[0]
                    TRAIN_FILES.append(sample)
        print('ID {} contains {} gloms.'.format(id, len(TRAIN_FILES)))
    MAX_COUNT += len(TRAIN_FILES)
    PAT_COUNT += len(ID_LIST[int(i * len(ID_LIST)//FOLDS):int((i+1) * len(ID_LIST)//FOLDS)])
    print('Fold {} contains {} images from {} patients, {} patients processed.'
          .format(i, len(TRAIN_FILES), len(ID_LIST[int(i * len(ID_LIST)//FOLDS):int((i+1) * len(ID_LIST)//FOLDS)]),
                  PAT_COUNT))
    TRAIN_FILES = np.array(TRAIN_FILES)
    np.savez('/home/cougarnet.uh.edu/srizvi7/Desktop/amr_fold{}'.format(i),
             FILES=TRAIN_FILES[:, 0], L1=TRAIN_FILES[:, 1], L1T=TRAIN_FILES[:, 2], L2=TRAIN_FILES[:, 3],
             L2T=TRAIN_FILES[:, 4], L3=TRAIN_FILES[:, 5], L3T=TRAIN_FILES[:, 6], L4=TRAIN_FILES[:, 7],
             L4T=TRAIN_FILES[:, 8], CONS=TRAIN_FILES[:, 9], AT=TRAIN_FILES[:, 10], AGR=TRAIN_FILES[:, 11]
             )

if MAX_COUNT == len(ALL_IMG) and PAT_COUNT == len(ID_LIST):
    print('Done. {} images split into {} folds, {} patients processed.'.format(MAX_COUNT, FOLDS, PAT_COUNT))
else:
    print('ERROR: Patient data has not been properly processed. Check patient and image counts. '
          'Tot images {} and tot cases {}'.format(MAX_COUNT, PAT_COUNT))