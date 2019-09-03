from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json

import numpy as np
import pandas as pd

import scipy.sparse as sparse
import data_utils as data
import datasets
import upper_bounds
import defenses
from upper_bounds import hinge_loss, hinge_grad, logistic_grad

attack_label = 'alfa'
percentile = 90
use_train = False
epsilons = [0.005, 0.01, 0.015, 0.02, 0.03]

for dataset_name in ['enron', 'mnist_17']:
    if dataset_name == 'enron':
        weight_decay = 0.09
    elif dataset_name == 'mnist_17':
        weight_decay = 0.01

    X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

    if use_train:
        raise NotImplementedError
    else:
        feasible_flipped_mask = defenses.find_feasible_label_flips_in_sphere(X_test, Y_test, percentile)
        X_augmented = data.vstack(X_train, X_test[feasible_flipped_mask, :])
        Y_augmented = data.vstack(Y_train, -Y_test[feasible_flipped_mask])

    print('X_train size: ', X_train.shape)
    print('X_augmented size: ', X_augmented.shape)

    n = X_train.shape[0]
    m = X_augmented.shape[0] - n
    X_flipped = X_augmented[n:, :]
    Y_flipped = Y_augmented[n:]

    gurobi_svm = upper_bounds.GurobiSVM(weight_decay=weight_decay)
    gurobi_svm.fit(X_train, Y_train, verbose=True)
    orig_losses = gurobi_svm.get_indiv_hinge_losses(X_flipped, Y_flipped)

    for epsilon in epsilons:
        print('>> epsilon %s' % epsilon)
        num_points_to_add = int(np.round(epsilon * n))
        q_finder = upper_bounds.QFinder(m=m, q_budget=num_points_to_add)
        q = np.ones(m) * (num_points_to_add / m)

        for iter_idx in range(100):
            old_q = q
            sample_weights = np.concatenate((
                np.ones(n),
                q))
            gurobi_svm.fit(X_augmented, Y_augmented, sample_weights=sample_weights, verbose=True)
            poisoned_losses = gurobi_svm.get_indiv_hinge_losses(X_flipped, Y_flipped)
            loss_diffs = poisoned_losses - orig_losses
            q = q_finder.solve(loss_diffs, verbose=True)
            print("At iteration %s, q is:" % iter_idx)
            print(q)
            if np.all(old_q == q):
                print('Done, terminating')
                break

        q_idx = np.where(q)[0][0]
        assert q[q_idx] == num_points_to_add
        if sparse.issparse(X_flipped):
            x = X_flipped[q_idx, :].toarray()
        else:
            x = X_flipped[q_idx, :]
        X_modified, Y_modified = data.add_points(
            x,
            Y_flipped[q_idx],
            X_train,
            Y_train,
            num_copies=num_points_to_add
        )

        attack_save_path = datasets.get_target_attack_npz_path(
            dataset_name,
            epsilon,
            weight_decay,
            percentile,
            attack_label)

        if sparse.issparse(X_modified):
            X_poison = X_modified[n:, :].asfptype()
        else:
            X_poison = X_modified[n:, :]

        np.savez(
            attack_save_path,
            X_poison=X_poison,
            Y_poison=Y_modified[n:]
            )
