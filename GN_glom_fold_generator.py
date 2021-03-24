import glob
import random
import numpy as np
import pandas as pd

# ------------------------------
# ------- FUNCTIONS ------------
# ------------------------------

def getSizeOfNestedList(listOfElem):
    ''' Get number of elements in a nested list '''

    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count

# ------------------------------
# ------- PREPROCESSING --------
# ------------------------------

# Path to the images for each data center
GLOM_PATH_COLOGNE = '/data/public/HULA/Cologne_GN/img/'
GLOM_PATH_SZEGED = '/data/public/HULA/Szeged_GN/img/'

# Getting the file names without directory
ALL_COLOGNE_IMG = glob.glob(GLOM_PATH_COLOGNE + '*.png')
ALL_COLOGNE_IMG = [s.replace('/data/public/HULA/Cologne_GN/img/', '') for s in ALL_COLOGNE_IMG]
ALL_SZEGED_IMG = glob.glob(GLOM_PATH_SZEGED + '*.png')
ALL_SZEGED_IMG = [s.replace('/data/public/HULA/Szeged_GN/img/', '') for s in ALL_SZEGED_IMG]

# Combining all of the file names for all centers
ALL_IMG = ALL_COLOGNE_IMG + ALL_SZEGED_IMG

# Loading the case info for each data center
COLOGNE_CASE_INFO = pd.read_csv('/data/public/HULA/Cologne_GN/CologneGNBiopsyList.csv')
SZEGED_CASE_INFO = pd.read_csv('/data/public/HULA/Szeged_GN/SzegedGNBiopsyList.csv')

# Combining, sorting the info for each data center
CASE_INFO_PRE = pd.concat([COLOGNE_CASE_INFO, SZEGED_CASE_INFO])
CASE_INFO_PRE = CASE_INFO_PRE.sort_values(by=['Houston'])

# Check to see if all crops can be linked to a case name
print('Testing for consistency between the files and the datasheets...')
failures = 0
for files in ALL_IMG:
    counts = 0
    cases = []
    for names in CASE_INFO_PRE['Local identifier']:
        if (names in files) and (len(files.split(names)[0]) == 0):
            counts += 1
            cases.append(names)
    if counts == 0:
        print('Warning: {} cant be linked to a case. Please check file name designations'.format(files))
        failures += 1
    elif counts > 1:
        print('Warning: {} is linked to more than one case. Please check file name designations'.format(files))
        failures += 1

if failures == 0:
    print('TEST SUCCESS: All images are associated to a single case.')
else:
    print('TEST FAIL: There are inconsistencies between the datasheets and available data!')

# Remove cases that have no annotations, printing their info, generating dataset summary sheet
CASE_INFO = CASE_INFO_PRE
CASE_STATS = pd.DataFrame(columns=['Center', 'Local Identifier', 'Case Label', 'Glomerulus Annotations',
                                    'Arteriole Annotations', 'Artery Annotations', 'Tubule Annotations',
                                   'Ignore Annotations', 'None Annotations'])
print('-----------------------------------------------')
for names in CASE_INFO_PRE['Local identifier']:
    counts = 0
    glom_counts = 0
    arteriole_counts = 0
    artery_counts = 0
    tubule_counts = 0
    ignore_counts = 0
    none_counts = 0
    for files in ALL_IMG:
        if (names in files) and (len(files.split(names)[0]) == 0):
            counts += 1
            if '_Glomerulus_' in files:
                glom_counts += 1
            elif '_Arteriole_' in files:
                arteriole_counts += 1
            elif '_Artery_' in files:
                artery_counts += 1
            elif '_Tubule_' in files:
                tubule_counts += 1
            elif '_Ignore*_' in files:
                ignore_counts += 1
            elif '_None_' in files:
                none_counts += 1
                print('WARNING: {} has not been assigned a compartment label. Please revise QuPath project.'.format(files))
    if counts == 0:
        print('Missing annotations for ' + names + ', skipping this case, its case label is ' + CASE_INFO[CASE_INFO['Local identifier'] == names]['Houston'].values[0])
        CASE_INFO = CASE_INFO[CASE_INFO['Local identifier'] != names]
    else:
        case_stat = [CASE_INFO[CASE_INFO['Local identifier'] == names]['Centre'].values[0],
                     names, CASE_INFO[CASE_INFO['Local identifier'] == names]['Houston'].values[0],
                     glom_counts, arteriole_counts, artery_counts, tubule_counts, ignore_counts, none_counts]
        CASE_STATS.loc[len(CASE_STATS)] = case_stat
print('-----------------------------------------------')

# Saving our global dataset statistics
CASE_STATS.to_csv(r'/data/public/HULA/dataset_stats/PanGN.csv')

# ------------------------------
# ----- FOLD GENERATION --------
# ------------------------------

# Set the desired number of train-val folds
FOLDS = 5

# Generating the scarce fold for testing (This applies to any class that is too scarce to train, optional)
# Generate the scarce info dataframe here, separate the scarce classes with or statements
SCARCE_INFO = CASE_INFO.loc[(CASE_INFO['Houston'] == 'C3-GN') | (CASE_INFO['Houston'] == 'CryoglobulinemicGN')]
SCARCE_FILES = []
SCARCE_PROBS = []
SCARCE_ID = []
SCARCE_LABELS = []
for id in SCARCE_INFO['Local identifier']:
    case_holder = []
    for file in ALL_IMG:
        if (id in file) and (len(file.split(id)[0]) == 0):
            case_holder.append(file)
    if SCARCE_INFO[SCARCE_INFO['Local identifier'] == id]['Houston'].values[0] == 'C3-GN':
        SCARCE_LABELS.append('C3-GN')
    elif SCARCE_INFO[SCARCE_INFO['Local identifier'] == id]['Houston'].values[0] == 'CryoglobulinemicGN':
        SCARCE_LABELS.append('CryoglobulinemicGN')
    scarce_probs = [1/len(case_holder)] * len(case_holder)
    SCARCE_FILES.append(case_holder)
    SCARCE_PROBS.append(scarce_probs)
    SCARCE_ID.append(id)

print('Number of scarce cases: ' + str(len(SCARCE_FILES)))
print('Number of scarce image samples: ' + str(getSizeOfNestedList(SCARCE_FILES)))

# Saving the file
np.savez('/data/public/HULA/folds/GN_all/GN_SCARCE.npz', FILES=SCARCE_FILES, CL=SCARCE_LABELS, ID=SCARCE_ID, PROBS=SCARCE_PROBS)

# Generating our train/validation folds on the rest of our data
CASE_INFO = CASE_INFO.loc[(CASE_INFO['Houston'] != 'C3-GN') & (CASE_INFO['Houston'] != 'CryoglobulinemicGN')]
CASE_FILES = []
CASE_PROBS = []
CASE_LABELS = []
CASE_ID = []

# Generate an embedded list for each of the folds
for x in range(FOLDS):
    CASE_FILES.append([])
    CASE_PROBS.append([])
    CASE_LABELS.append([])
    CASE_ID.append([])

# Generate the folds
for classes in CASE_INFO.Houston.unique().tolist():
    case_names = CASE_INFO[CASE_INFO['Houston'] == classes]['Local identifier'].to_list()
    random.shuffle(case_names)
    for i in range(FOLDS+1):
        if (i == FOLDS) and ((len(case_names) % FOLDS) != 0):
            last_cases = case_names[(len(case_names)-len(case_names) % FOLDS):len(case_names)]
            for case in last_cases:
                fold_counts = []
                case_holder = []
                for j in range(FOLDS):
                    fold_counts.append(getSizeOfNestedList(CASE_FILES[j]))
                for file in ALL_IMG:
                    if (case in file) and (len(file.split(case)[0]) == 0):
                        case_holder.append(file)
                case_probs = [1/len(case_holder)] * len(case_holder)
                CASE_PROBS[fold_counts.index(min(fold_counts))].append(case_probs)
                CASE_FILES[fold_counts.index(min(fold_counts))].append(case_holder)
                CASE_LABELS[fold_counts.index(min(fold_counts))].append(CASE_INFO[CASE_INFO['Local identifier'] == case]['Houston'].values[0])
                CASE_ID[fold_counts.index(min(fold_counts))].append(case)
        elif (i == FOLDS) and ((len(case_names) % FOLDS) == 0):
            break
        else:
            fold_cases = case_names[int(i * (len(case_names)//FOLDS)):int((i+1) * (len(case_names)//FOLDS))]
            for case in fold_cases:
                case_holder = []
                for file in ALL_IMG:
                    if (case in file) and (len(file.split(case)[0]) == 0):
                        case_holder.append(file)
                case_probs = [1/len(case_holder)] * len(case_holder)
                CASE_PROBS[i].append(case_probs)
                CASE_FILES[i].append(case_holder)
                CASE_LABELS[i].append(CASE_INFO[CASE_INFO['Local identifier'] == case]['Houston'].values[0])
                CASE_ID[i].append(case)

# Generate fold files, print final stats
count_img = 0
count_case = 0
for j in range(len(CASE_FILES)):
    print('Number of cases, fold {}: '.format(j) + str(len(CASE_FILES[j])))
    print('Number of image samples, fold {}: '.format(j) + str(getSizeOfNestedList(CASE_FILES[j])))
    np.savez('/data/public/HULA/folds/GN_all/GN_FOLD_{}.npz'.format(j), FILES=CASE_FILES[j], CL=CASE_LABELS[j], ID=CASE_ID[j], PROBS=CASE_PROBS[j])
    count_img += getSizeOfNestedList(CASE_FILES[j])
    count_case += len(CASE_FILES[j])
print('Total number of cases: ' + str(count_case))
print('Total number of images: ' + str(count_img))

