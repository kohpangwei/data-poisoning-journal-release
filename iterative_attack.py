from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import numpy as np

import os
import defenses
import data_utils as data

def poison_with_influence_proj_gradient_step(model, test_idx, indices_to_poison,
    projection_fn,
    step_size=0.01,
    shrink_towards='cluster_center',
    loss_type='normal_loss',
    force_refresh=True,
    test_description=None,
    output_root=None):
    """
    Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    that has been modified by a single gradient step.
    """

    data_sets = model.data_sets

    if test_description is None:
        test_description = test_idx
    grad_filename = os.path.join(output_root, 'grad_influence_wrt_input_val_%s_testidx_%s.npy' % (model.model_name, test_description))

    if (force_refresh == False) and (os.path.exists(grad_filename)):
        grad_influence_wrt_input_val = np.load(grad_filename)
    else:
        grad_influence_wrt_input_val = model.get_grad_of_influence_wrt_input(
            indices_to_poison,
            test_idx,
            verbose=False,
            force_refresh=force_refresh,
            test_description=test_description,
            loss_type=loss_type)

    poisoned_X_train = data_sets.train.x[indices_to_poison, :]
    poisoned_X_train -= step_size * grad_influence_wrt_input_val

    poisoned_labels = data_sets.train.labels[indices_to_poison]

    weights = model.sess.run(model.weights)
    print('weights shape is ', weights.shape)
    poisoned_X_train = projection_fn(
        poisoned_X_train,
        poisoned_labels,
        theta=weights[:-1],
        bias=weights[-1])

    return poisoned_X_train


def iterative_attack(
    model,
    indices_to_poison,
    test_idx,
    test_description=None,
    step_size=0.01,
    num_iter=10,
    loss_type='normal_loss',
    projection_fn=None,
    output_root=None,
    num_copies=None,
    stop_after=3,
    start_time=None):

    if num_copies is not None:
        assert len(num_copies) == 2
        assert np.min(num_copies) >= 1
        assert len(indices_to_poison) == 2
        assert indices_to_poison[1] == (indices_to_poison[0] + 1)
        assert indices_to_poison[1] + num_copies[0] + num_copies[1] == (model.data_sets.train.x.shape[0] - 1)
        assert model.data_sets.train.labels[indices_to_poison[0]] == 1
        assert model.data_sets.train.labels[indices_to_poison[1]] == -1
        copy_start = indices_to_poison[1] + 1
        assert np.all(model.data_sets.train.labels[copy_start:copy_start+num_copies[0]] == 1)
        assert np.all(model.data_sets.train.labels[copy_start+num_copies[0]:copy_start+num_copies[0]+num_copies[1]] == -1)

    largest_test_loss = 0
    stop_counter = 0

    print('Test idx: %s' % test_idx)

    if start_time is not None:
        assert num_copies is not None
        times_taken = np.zeros(num_iter)
        Xs_poison = np.zeros((num_iter, len(indices_to_poison), model.data_sets.train.x.shape[1]))
        Ys_poison = np.zeros((num_iter, len(indices_to_poison)))
        nums_copies = np.zeros((num_iter, len(indices_to_poison)))

    for attack_iter in range(num_iter):
        print('*** Iter: %s' % attack_iter)

        # Create modified training dataset
        old_poisoned_X_train = np.copy(model.data_sets.train.x[indices_to_poison, :])

        poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
            model,
            test_idx,
            indices_to_poison,
            projection_fn,
            step_size=step_size,
            loss_type=loss_type,
            force_refresh=True,
            test_description=test_description,
            output_root=output_root)

        if num_copies is not None:
            poisoned_X_train = model.data_sets.train.x
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset
            copy_start = indices_to_poison[1] + 1
            poisoned_X_train[copy_start:copy_start+num_copies[0], :] = poisoned_X_train_subset[0, :]
            poisoned_X_train[copy_start+num_copies[0]:copy_start+num_copies[0]+num_copies[1], :] = poisoned_X_train_subset[1, :]
        else:
            poisoned_X_train = np.copy(model.data_sets.train.x)
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset

        # Measure some metrics on what the gradient step did
        labels = model.data_sets.train.labels
        dists_sum = 0.0
        poisoned_dists_sum = 0.0
        poisoned_mask = np.array([False] * len(labels), dtype=bool)
        poisoned_mask[indices_to_poison] = True
        for y in set(labels):
            cluster_center = np.mean(poisoned_X_train[labels == y, :], axis=0)
            dists = np.linalg.norm(poisoned_X_train[labels == y, :] - cluster_center, axis=1)
            dists_sum += np.sum(dists)

            poisoned_dists = np.linalg.norm(poisoned_X_train[(labels == y) & (poisoned_mask), :] - cluster_center, axis=1)
            poisoned_dists_sum += np.sum(poisoned_dists)

        dists_mean = dists_sum / len(labels)
        poisoned_dists_mean = poisoned_dists_sum / len(indices_to_poison)

        dists_moved = np.linalg.norm(old_poisoned_X_train - poisoned_X_train[indices_to_poison, :], axis=1)
        print('Average distance to cluster center (overall): %s' % dists_mean)
        print('Average distance to cluster center (poisoned): %s' % poisoned_dists_mean)
        print('Average diff in X_train among poisoned indices = %s' % np.mean(dists_moved))
        print('Fraction of 0 gradient points: %s' % np.mean(dists_moved == 0))
        print('Average distance moved by points that moved: %s' % np.mean(dists_moved[dists_moved > 0]))

        # Update training dataset
        model.update_train_x(poisoned_X_train)

        # Retrain model
        results = model.train()

        if start_time is not None:
            end_time = time.time()
            times_taken[attack_iter] = end_time - start_time
            Xs_poison[attack_iter, :, :] = np.copy(poisoned_X_train_subset)
            Ys_poison[attack_iter, :] = model.data_sets.train.labels[indices_to_poison]
            nums_copies[attack_iter, :] = num_copies

        print('attack_iter', attack_iter)
        print('num_iter - 1', num_iter - 1)

        if ((attack_iter + 1) % 10 == 0) or (attack_iter == num_iter - 1):
            print('in')
            # Calculate test loss
            test_loss = results['test_loss']
            if largest_test_loss < test_loss:
                print('test loss match')
                largest_test_loss = test_loss

                np.savez(os.path.join(output_root, '%s_attack' % (model.model_name)),
                    poisoned_X_train=poisoned_X_train,
                    Y_train=model.data_sets.train.labels,
                    attack_iter=attack_iter + 1)

                stop_counter = 0
            else:
                stop_counter += 1

            if start_time is not None:
                np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
                    times_taken=times_taken,
                    nums_copies=nums_copies)

            if stop_counter >= stop_after:
                break

    if start_time is not None:
        np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
            times_taken=times_taken,
            Xs_poison=Xs_poison,
            Ys_poison=Ys_poison,
            nums_copies=nums_copies)


def get_feasible_flipped_mask(
    X_train, Y_train,
    centroids,
    centroid_vec,
    sphere_radii,
    slab_radii,
    class_map,
    use_slab=False):

    sphere_dists_flip = defenses.compute_dists_under_Q(
        X_train, -Y_train,
        Q=None,
        subtract_from_l2=False,
        centroids=centroids,
        class_map=class_map,
        norm=2)

    if use_slab:
        slab_dists_flip = defenses.compute_dists_under_Q(
            X_train, -Y_train,
            Q=centroid_vec,
            subtract_from_l2=False,
            centroids=centroids,
            class_map=class_map,
            norm=2)

    feasible_flipped_mask = np.zeros(X_train.shape[0], dtype=bool)

    for y in set(Y_train):
        class_idx_flip = class_map[-y]
        sphere_radius_flip = sphere_radii[class_idx_flip]

        feasible_flipped_mask[Y_train == y] = (sphere_dists_flip[Y_train == y] <= sphere_radius_flip)

        if use_slab:
            slab_radius_flip = slab_radii[class_idx_flip]
            feasible_flipped_mask[Y_train == y] = (
                feasible_flipped_mask[Y_train == y] &
                (slab_dists_flip[Y_train == y] <= slab_radius_flip))

    return feasible_flipped_mask


def init_gradient_attack_from_mask(
    X_train, Y_train,
    epsilon,
    feasible_flipped_mask,
    use_copy=True):

    if not use_copy:
        num_copies = int(np.round(epsilon * X_train.shape[0]))

        idx_to_copy = np.random.choice(
            np.where(feasible_flipped_mask)[0],
            size=num_copies,
            replace=True)

        X_modified = data.vstack(X_train, X_train[idx_to_copy, :])
        Y_modified = np.append(Y_train, -Y_train[idx_to_copy])
        copy_array = None
        indices_to_poison = np.arange(X_train.shape[0], X_modified.shape[0])

    else:
        num_copies = int(np.round(epsilon * X_train.shape[0]))
        # Choose this in inverse class balance
        num_pos_copies = int(np.round(np.mean(Y_train == -1) * num_copies))
        num_neg_copies = num_copies - num_pos_copies

        np.random.seed(0)
        pos_idx_to_copy = np.random.choice(
            np.where(feasible_flipped_mask & (Y_train == -1))[0])
        neg_idx_to_copy = np.random.choice(
            np.where(feasible_flipped_mask & (Y_train == 1))[0])

        num_pos_copies -= 1
        num_neg_copies -= 1

        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_train,
            Y_train,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_modified,
            Y_modified,
            num_copies=num_pos_copies)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=num_neg_copies)
        copy_array = [num_pos_copies, num_neg_copies]
        indices_to_poison = np.arange(X_train.shape[0], X_train.shape[0]+2)

    return X_modified, Y_modified, indices_to_poison, copy_array
