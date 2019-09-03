from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import argparse
import time

import numpy as np

import data_utils as data
import datasets
import upper_bounds
import defenses
from upper_bounds import hinge_loss, hinge_grad, logistic_grad

def kkt_setup(
    target_theta,
    target_bias,
    X_train,
    Y_train,
    X_test,
    Y_test,
    dataset_name,
    percentile,
    loss_percentile,
    model,
    model_grad,
    class_map,
    use_slab,
    use_loss):

    clean_grad_at_target_theta, clean_bias_grad_at_target_theta = model_grad(
        target_theta,
        target_bias,
        X_train,
        Y_train)

    losses_at_target = upper_bounds.indiv_hinge_losses(
        target_theta,
        target_bias,
        X_train,
        Y_train)

    sv_indices = losses_at_target > 0

    _, sv_centroids, _, sv_sphere_radii, _ = data.get_data_params(
        X_train[sv_indices, :],
        Y_train[sv_indices],
        percentile=percentile)

    max_losses = [0, 0]
    for y in set(Y_train):
        max_losses[class_map[y]] = np.percentile(losses_at_target[Y_train == y], loss_percentile)

    print('Max losses are: %s' % max_losses)
    model.coef_ = target_theta.reshape((1, -1))
    model.intercept_ = target_bias

    print('If we could get our targeted theta exactly:')
    print('Train            : %.3f' % model.score(X_train, Y_train))
    print('Test (overall)   : %.3f' % model.score(X_test, Y_test))


    two_class_kkt = upper_bounds.TwoClassKKT(
        clean_grad_at_target_theta.shape[0],
        dataset_name=dataset_name,
        X=X_train,
        use_slab=use_slab,
        constrain_max_loss=use_loss)

    target_bias_grad = clean_bias_grad_at_target_theta

    return two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses

def kkt_attack(two_class_kkt,
               target_grad, target_theta,
               total_epsilon, epsilon_pos, epsilon_neg,
               X_train, Y_train,
               class_map, centroids, centroid_vec, sphere_radii, slab_radii,
               target_bias, target_bias_grad, max_losses,
               sv_centroids=None, sv_sphere_radii=None):

    x_pos, x_neg, epsilon_pos, epsilon_neg = two_class_kkt.solve(
        target_grad,
        target_theta,
        epsilon_pos,
        epsilon_neg,
        class_map,
        centroids,
        centroid_vec,
        sphere_radii,
        slab_radii,
        target_bias=target_bias,
        target_bias_grad=target_bias_grad,
        max_losses=max_losses,
        verbose=False)

    obj = np.linalg.norm(target_grad - epsilon_pos * x_pos.reshape(-1) + epsilon_neg * x_neg.reshape(-1))
    print("** Actual objective value: %.4f" % obj)
    num_train = X_train.shape[0]
    total_points_to_add = int(np.round(total_epsilon * X_train.shape[0]))
    num_pos = int(np.round(epsilon_pos * X_train.shape[0]))
    num_neg = total_points_to_add - num_pos
    assert num_neg >= 0

    X_modified, Y_modified = data.add_points(
        x_pos,
        1,
        X_train,
        Y_train,
        num_copies=num_pos)
    X_modified, Y_modified = data.add_points(
        x_neg,
        -1,
        X_modified,
        Y_modified,
        num_copies=num_neg)

    return X_modified, Y_modified, obj, x_pos, x_neg, num_pos, num_neg
