from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import shutil
import argparse

import numpy as np

import defenses
import datasets
import data_utils as data

batch_size = 100

parser = argparse.ArgumentParser()
parser.add_argument('attack_label', help='Label')
parser.add_argument('--repeat_points', default=1)

args = parser.parse_args()

attack_label = args.attack_label
repeat_points = int(np.round(float(args.repeat_points)))

for dataset_name in [
    'enron',
    # 'mnist_17',
    # 'dogfish'
    ]:

    if dataset_name == 'enron':
        weight_decay = '0.09'
    elif dataset_name == 'mnist_17':
        weight_decay = '0.01'
    elif dataset_name == 'dogfish':
        weight_decay = '1.1'
    else:
        break

    dataset_folder = os.path.join(datasets.OUTPUT_FOLDER, dataset_name)
    grad_folder = os.path.join(dataset_folder, 'influence_data')

    prefix = 'smooth_hinge_%s_sphere-True_slab-False_start-copy_lflip-True_step-0.1_t-0.001_eps-' % dataset_name
    suffix = '_wd-%s_rs-1_' % weight_decay
    if attack_label == 'influence-standard':
        suffix += 'attack.npz'

    elif attack_label == 'influence-copy-standard':
        suffix += 'copy_attack.npz'

    elif attack_label == 'influence-copy-train':
        suffix += 'train_copy_attack.npz'

    elif attack_label == 'influence-copy-standard-flip':
        suffix += 'copy_flip-loss_attack.npz'

    elif attack_label == 'influence-baseline':
        prefix = 'smooth_hinge_%s_sphere-True_slab-False_start-copy_lflip-True_step-0.1_t-0_eps-' % dataset_name
        if dataset_name == 'enron':
            suffix += 'percentile-80_baseline_attack.npz'
        elif dataset_name == 'mnist_17':
            suffix += 'no-smooth_attack.npz'
        else:
            print('skipping %s for %s' % (dataset_name, attack_label))
            continue

    elif attack_label == 'influence-noLP':
        if dataset_name == 'enron':
            suffix += 'percentile-80_copy_no-LP_attack.npz'
        else:
            print('skipping %s for %s' % (dataset_name, attack_label))
            continue

    elif attack_label == 'influence-em-5000-baseline':
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.1_t-0_eps-' % dataset_name
        if dataset_name == 'enron':
            suffix += 'percentile-80_em-5000_baseline_attack.npz'
        else:
            print('skipping %s for %s' % (dataset_name, attack_label))
            continue
        print(suffix)

    elif attack_label == 'influence-em-5000-baseline-smooth':
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.1_t-0.001_eps-' % dataset_name
        if dataset_name == 'enron':
            suffix += 'percentile-80_em-5000_baseline-smooth_attack.npz'
        else:
            print('skipping %s for %s' % (dataset_name, attack_label))
            continue

    elif attack_label == 'influence-em-5000-noLP':
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.1_t-0.001_eps-' % dataset_name
        if dataset_name == 'enron':
            suffix += 'percentile-80_em-5000_no-LP_attack.npz'
        else:
            print('skipping %s for %s' % (dataset_name, attack_label))
            continue
        print(suffix)

    elif 'influence-em-0.01-' in attack_label:
        # Slab is true
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.01_t-0.001_eps-' % dataset_name

        suffix += 'em-%s_attack.npz' % attack_label.split('influence-em-0.01-')[1]
        print(suffix)

    elif 'influence-em-0-usetrain' in attack_label:
        # Slab is true
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.1_t-0.001_eps-' % dataset_name

        suffix += 'em-0_use-train_attack.npz'
        print(suffix)

    elif 'influence-em-' in attack_label:
        # Slab is true
        prefix = 'smooth_hinge_%s_sphere-True_slab-True_start-copy_lflip-True_step-0.1_t-0.001_eps-' % dataset_name

        suffix += 'em-%s_attack.npz' % attack_label.split('influence-em-')[1]
        print(suffix)

    if repeat_points > 1:
        attack_label = attack_label + '-repeat%s' % repeat_points

    X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

    for grad_filename in os.listdir(grad_folder):
        if grad_filename.startswith(prefix) and grad_filename.endswith(suffix):

            epsilon = grad_filename.split('_eps-')[1].split('_')[0]
            target_filename = '%s_attack_wd-%s_percentile-90_epsilon-%s_label-%s.npz' % (
                dataset_name,
                weight_decay,
                epsilon,
                attack_label)

            print(target_filename)
            grad_path = os.path.join(grad_folder, grad_filename)
            target_path = os.path.join(dataset_folder, target_filename)

            f = np.load(grad_path)

            actual_num_train = X_train.shape[0]

            X_poison = f['poisoned_X_train'][actual_num_train:, :]
            Y_poison = f['Y_train'][actual_num_train:]

            print(np.min(X_poison), np.max(X_poison))

            if dataset_name in ['enron', 'mnist_17']:
                assert (np.min(X_poison) >= -0.05)
                X_poison[X_poison < 0] = 0

            if dataset_name == 'enron':
                X_poison_round = data.rround_with_repeats(X_poison, Y_poison, repeat_points)
            else:
                X_poison_round = X_poison

            np.savez(
                target_path,
                X_poison=X_poison_round,
                Y_poison=Y_poison)
