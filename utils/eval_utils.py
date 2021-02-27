import torch
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.utils.multiclass import unique_labels
import os

def get_binary_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def compute_accuracy(target, output, classes):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    if classes == 2:
        accuracy = get_binary_accuracy(target, output.squeeze(1))
    else:
        num_correct = torch.sum(target == torch.argmax(output, dim=1))
        accuracy = num_correct.float() / num_samples
    return accuracy

def mutual_info(mc_prob):
    """
    computes the mutual information
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, num_cls]
    :return: mutual information of shape [batch_size, num_cls]
    """
    eps = 1e-5
    mean_prob = mc_prob.mean(axis=0)
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term


def predictive_entropy(prob):
    """
    Entropy of the probabilities (to measure the epistemic uncertainty)
    :param prob: probabilities of shape [batch_size, C]
    :return: Entropy of shape [batch_size]
    """
    eps = 1e-5
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)

def save_confusion_matrix(y_true, y_pred, classes, dest_path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    # This function plots and saves the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(dest_path)"""


def uncertainty_fraction_removal(y, y_pred, y_var, num_fracs, num_random_reps, save=False, save_dir=''):
    fractions = np.linspace(1 / num_fracs, 1, num_fracs)
    num_samples = y.shape[0]
    acc_unc_sort = np.array([])
    acc_pred_sort = np.array([])
    acc_random_frac = np.zeros((0, num_fracs))

    remain_samples = []
    # uncertainty-based removal
    inds = y_var.argsort()
    y_sorted = y[inds]
    y_pred_sorted = y_pred[inds]
    for frac in fractions:
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples.append(len(y_temp))
        y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
        acc_unc_sort = np.append(acc_unc_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    # random removal
    for rep in range(num_random_reps):
        acc_random_sort = np.array([])
        perm = np.random.permutation(y_var.shape[0])
        y_sorted = y[perm]
        y_pred_sorted = y_pred[perm]
        for frac in fractions:
            y_temp = y_sorted[:int(num_samples * frac)]
            y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
            acc_random_sort = np.append(acc_random_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        acc_random_frac = np.concatenate((acc_random_frac, np.reshape(acc_random_sort, [1, -1])), axis=0)
    acc_random_m = np.mean(acc_random_frac, axis=0)
    acc_random_s = np.std(acc_random_frac, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')

    line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
    ax.fill_between(fractions,
                    acc_random_m - acc_random_s,
                    acc_random_m + acc_random_s,
                    color='black', alpha=0.3)
    line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.set_xlabel('Fraction of Retained Data')
    ax.set_ylabel('Prediction Accuracy')
    if save:
        plt.savefig(os.path.join(save_dir, 'uncertainty_fraction_removal.svg'))
    return acc_unc_sort, acc_random_frac

def combo_uncertainty_fraction_removal(y, y_pred, y_var, aug_pred, aug_var, num_fracs, num_random_reps, save=False, save_dir=''):
    fractions = np.linspace(1 / num_fracs, 1, num_fracs)
    num_samples = y.shape[0]
    acc_unc_sort = np.array([])
    acc_pred_sort = np.array([])
    acc_random_frac = np.zeros((0, num_fracs))

    remain_samples = []
    # uncertainty-based removal (baseline)
    inds = y_var.argsort()
    y_sorted = y[inds]
    y_pred_sorted = y_pred[inds]
    for frac in fractions:
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples.append(len(y_temp))
        y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
        acc_unc_sort = np.append(acc_unc_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    # augmented unc based removal
    acc_unc_sort_aug = np.array([])
    acc_pred_sort_aug = np.array([])
    acc_random_frac_aug = np.zeros((0, num_fracs))

    remain_samples_aug = []
    # uncertainty-based removal
    aug_inds = aug_var.argsort()
    y_sorted = y[aug_inds]
    aug_pred_sorted = aug_pred[aug_inds]
    for frac in fractions:
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples_aug.append(len(y_temp))
        aug_pred_temp = aug_pred_sorted[:int(num_samples * frac)]
        acc_unc_sort_aug = np.append(acc_unc_sort_aug, np.sum(y_temp == aug_pred_temp) / y_temp.shape[0])

    # random removal
    for rep in range(num_random_reps):
        acc_random_sort = np.array([])
        perm = np.random.permutation(y_var.shape[0])
        y_sorted = y[perm]
        y_pred_sorted = y_pred[perm]
        for frac in fractions:
            y_temp = y_sorted[:int(num_samples * frac)]
            y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
            acc_random_sort = np.append(acc_random_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        acc_random_frac = np.concatenate((acc_random_frac, np.reshape(acc_random_sort, [1, -1])), axis=0)
    acc_random_m = np.mean(acc_random_frac, axis=0)
    acc_random_s = np.std(acc_random_frac, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based_base', markersize=3, color='royalblue')
    ax.plot(fractions, acc_unc_sort_aug, '-v', lw=1.5, label='uncertainty-based_aug', markersize=3, color='red')

    line1, = ax.plot(fractions, acc_random_m, '-^', lw=1, label='Random', markersize=3, color='black')
    ax.fill_between(fractions,
                    acc_random_m - acc_random_s,
                    acc_random_m + acc_random_s,
                    color='black', alpha=0.3)
    line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.set_xlabel('Fraction of Retained Data')
    ax.set_ylabel('Prediction Accuracy')
    if save:
        plt.savefig(os.path.join(save_dir, 'uncertainty_fraction_removal_combo.svg'))


def normalized_uncertainty_toleration_removal(y, y_pred, y_var, num_points, save=False, save_dir=''):
    acc_uncertainty, acc_overall = np.array([]), np.array([])
    num_cls = len(np.unique(y))
    y_var = (y_var - y_var.min()) / (y_var.max() - y_var.min())
    per_class_remain_count = np.zeros((num_points, num_cls))
    per_class_acc = np.zeros((num_points, num_cls))
    thresholds = np.linspace(0, 1, num_points)
    remain_samples = []
    for i, t in enumerate(thresholds):
        idx = np.argwhere(y_var >= t)
        y_temp = np.delete(y, idx)
        remain_samples.append(len(y_temp))
        y_pred_temp = np.delete(y_pred, idx)
        acc_uncertainty = np.append(acc_uncertainty, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        if len(y_temp):
            per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
            per_class_acc[i, :] = np.array(
                [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
                 range(num_cls)])

    plt.figure()
    plt.plot(thresholds, acc_uncertainty, lw=1.5, color='royalblue', marker='o', markersize=4)
    plt.xlabel('Normalized Tolerated Model Uncertainty')
    plt.ylabel('Prediction Accuracy')
    if save:
        plt.savefig(os.path.join(save_dir, 'uncertainty_toleration_removal.png'))
    return(acc_uncertainty)