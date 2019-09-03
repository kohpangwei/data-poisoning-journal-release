from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import argparse
import time

import numpy as np

from sklearn import linear_model, svm

import scipy.io as sio
import data_utils as data
import datasets
import upper_bounds
import defenses
from upper_bounds import hinge_loss, hinge_grad, logistic_grad
import kkt_attack
import attack_utils


parser = argparse.ArgumentParser()
parser.add_argument('attack_type', help='One of: kkt-standard, kkt-loss, kkt-loss-slab, kkt-slab, kkt-baseline-slab')
parser.add_argument('--attack_label', default='', help='Label')
parser.add_argument('--dataset', default='enron')
parser.add_argument('--percentile', default=90)
parser.add_argument('--loss_percentile', default=90)
parser.add_argument('--repeat_points', default=1)
parser.add_argument('--use_train', action="store_true")
parser.add_argument('--no_round', action="store_true")
parser.add_argument('--timed', action="store_true")
parser.add_argument("--debug", help="Changes parameters so that everything is sped up.", action="store_true")

args = parser.parse_args()

attack_type = args.attack_type
attack_label = attack_type + args.attack_label
percentile = int(np.round(float(args.percentile)))
print('percentile is %s' % percentile)
loss_percentile = int(np.round(float(args.loss_percentile)))
print('loss percentile is %s' % loss_percentile)
repeat_points = int(np.round(float(args.repeat_points)))
dataset_name = args.dataset
use_train = args.use_train
no_round = args.no_round
debug = args.debug
timed = args.timed

if timed:
    start_time = time.time()
else:
    start_time = None

assert '-'.join(attack_type.split('-')[1:]) in ['standard', 'loss', 'loss-slab', 'slab', 'baseline-slab']

assert dataset_name in ['enron', 'mnist_17', 'dogfish', 'imdb']

if timed:
    epsilons = [0.03]
    epsilon_increment = 0.005
else:
    epsilons = [0.03]
    epsilon_increment = 0.005

if use_train:
    attack_label = attack_label + '-train'

if no_round:
    attack_label = attack_label + '-noround'

if timed:
    attack_label = attack_label + '-timed'

if repeat_points > 1:
    attack_label = attack_label + '-repeat%s' % repeat_points

if debug:
    epsilons = [0.03]
    num_iter_after_burnin = 1000
    print_interval = 30
    attack_label = attack_label + '-debug'

if dataset_name == 'enron':
    weight_decay = 0.09
    if use_train:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'enron_thetas_with_bias_exact_decay_09_use_train_v3.mat'))
    else:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'enron_thetas_with_bias_exact_decay_09_v3.mat'))
elif dataset_name == 'mnist_17':
    weight_decay = 0.01
    if use_train:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'mnist_17_thetas_with_bias_exact_decay_01_use_train_v3.mat'))
    else:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'mnist_17_thetas_with_bias_exact_decay_01_v3.mat'))
elif dataset_name == 'dogfish':
    weight_decay = 1.1
    if use_train:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'dogfish_thetas_with_bias_exact_decay_110_use_train_v3.mat'))
    else:
        f = sio.loadmat(
            os.path.join(
                datasets.DATA_FOLDER,
                'dogfish_thetas_with_bias_exact_decay_110_v3.mat_prune'))
elif dataset_name == 'imdb':
    weight_decay = 0.01
    f = sio.loadmat(
        os.path.join(
            datasets.DATA_FOLDER,
            'imdb_thetas_with_bias_exact_decay_01_all_v3_prune.mat'))

num_thetas = f['thetas'].shape[0]
num_features = f['thetas'][0][0].shape[0]
thetas = np.zeros((num_thetas, num_features))
biases = np.zeros(num_thetas)
for i in range(num_thetas):
    thetas[i, :] = f['thetas'][i][0][:, 0]
    biases[i] = f['biases'][i][0][0][0]

frac_to_remove = 0.05
max_iter = 1000

use_slab = 'slab' in attack_type
use_loss = 'loss' in attack_type
use_baseline = 'baseline' in attack_type

print('Frac to remove is: %s' % frac_to_remove)

def svm_model(**kwargs):
    return svm.LinearSVC(loss='hinge', **kwargs)

ScikitModel = svm_model
model_grad = hinge_grad
fit_intercept = True
X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)
if use_train: # Remove all information about test
    X_test = X_train
    Y_test = Y_train

# Then we calculate sphere radii at the 90th percentile to construct poisoned points
class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
    X_train,
    Y_train,
    percentile=percentile)

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)

C = 1.0 / (X_train.shape[0] * weight_decay)
model = ScikitModel(
    C=C,
    tol=1e-8,
    fit_intercept=fit_intercept,
    random_state=24,
    max_iter=max_iter,
    verbose=True)
model.fit(X_train, Y_train)

print()
print('Without any poisoning:')
print('Train            : %.3f' % model.score(X_train, Y_train))
print('Test (overall)   : %.3f' % model.score(X_test, Y_test))

print('Sanity check: these numbers should before near 0')
clean_grad_at_orig_theta, clean_bias_grad_at_orig_theta = model_grad(
    model.coef_.reshape(-1),
    model.intercept_,
    X_train,
    Y_train)
print(np.max(clean_grad_at_orig_theta + weight_decay * model.coef_.reshape(-1)))
print(np.max(clean_bias_grad_at_orig_theta))

### Loop over all target thetas to attack

best_attacks = {}
all_results = {}
for epsilon in epsilons:
    best_attacks[epsilon] = {}
    best_attacks[epsilon]['best_acc'] = None
    all_results[epsilon] = {}
    all_results[epsilon]['test acc before rounding'] = []
    all_results[epsilon]['test acc after rounding, before defense'] = []
    all_results[epsilon]['test acc after rounding, after defense'] = []
    all_results[epsilon]['theta_idx'] = []
    all_results[epsilon]['theta_distance'] = []
    all_results[epsilon]['bias_distance'] = []
    all_results[epsilon]['epsilon_pos'] = []
    all_results[epsilon]['epsilon_neg'] = []

if timed:
    assert len(epsilons) == 1
    num_attacks_to_allocate = 2000
    times_taken = np.zeros(num_attacks_to_allocate)
    Xs_poison = np.zeros((num_attacks_to_allocate, 2, X_train.shape[1]))
    Ys_poison = np.zeros((num_attacks_to_allocate, 2))
    nums_copies = np.zeros((num_attacks_to_allocate, 2))
    timer_idx = 0

perm = np.arange(f['thetas'].shape[0])
np.random.seed(0)
np.random.shuffle(perm)
if debug:
    if dataset_name == 'enron':
        perm[0] = 25
    elif dataset_name == 'imdb':
        perm[0] = 0

for perm_idx, theta_idx in enumerate(perm):

    print('\n>>>>>>>>>>>>>>>>>>> target theta %s (#%s)' % (theta_idx, perm_idx))
    target_theta = thetas[theta_idx, :]
    if fit_intercept:
        target_bias = biases[theta_idx]
    else:
        target_bias = 0

    two_class_kkt, clean_grad_at_target_theta, target_bias_grad, max_losses = kkt_attack.kkt_setup(
        target_theta,
        target_bias,
        X_train, Y_train,
        X_test, Y_test,
        dataset_name,
        percentile,
        loss_percentile,
        model,
        model_grad,
        class_map,
        use_slab,
        use_loss)

    for total_epsilon in epsilons:

        print('\n>>>>> total epsilon: %s' % total_epsilon)

        target_grad = clean_grad_at_target_theta + ((1 + total_epsilon) * weight_decay * target_theta)
        epsilon_pairs = []

        # If it's possible to choose epsilon_pos and epsilon_neg such that the bias is exact,
        # then try it.
        # We want: total_epsilon = epsilon_pos + epsilon_neg
        # and:     target_bias_grad - epsilon_pos + epsilon_neg = 0
        # which results in...
        epsilon_neg = (total_epsilon - target_bias_grad) / 2
        epsilon_pos = total_epsilon - epsilon_neg

        if (epsilon_neg >= 0) and (epsilon_neg <= total_epsilon):
            epsilon_pairs.append((epsilon_pos, epsilon_neg))

        for epsilon_pos in np.arange(0, total_epsilon + 1e-6, epsilon_increment):
            epsilon_neg = total_epsilon - epsilon_pos
            epsilon_pairs.append((epsilon_pos, epsilon_neg))

        for epsilon_pos, epsilon_neg in epsilon_pairs:
            print('\n## Trying epsilon_pos %s, epsilon_neg %s' % (epsilon_pos, epsilon_neg))

            X_modified, Y_modified, obj, x_pos, x, num_pos, num_neg = kkt_attack.kkt_attack(
                two_class_kkt,
                target_grad, target_theta,
                total_epsilon, epsilon_pos, epsilon_neg,
                X_train, Y_train,
                class_map, centroids, centroid_vec, sphere_radii, slab_radii,
                target_bias, target_bias_grad, max_losses)

            if timed:
                end_time = time.time()
                times_taken[timer_idx] = end_time - start_time
                print('Time taken so far: %s' % times_taken[timer_idx])

                Xs_poison[timer_idx, 0, :] = x_pos.reshape(-1)
                Ys_poison[timer_idx, 0] = 1
                nums_copies[timer_idx, 0] = num_pos

                Xs_poison[timer_idx, 1, :] = x_neg.reshape(-1)
                Ys_poison[timer_idx, 1] = -1
                nums_copies[timer_idx, 1] = num_neg

                timer_idx += 1

                # If timed, no need to test defenses; this makes it more fair w.r.t. influence
                continue

            X_modified_round = attack_utils.rround_if_needed(
                X_modified, Y_modified,
                X_train, Y_train,
                dataset_name,
                no_round,
                repeat_points)

            all_results, best_attacks = attack_utils.test_def_and_update_results(X_modified_round, Y_modified,
                X_train, Y_train,
                X_test, Y_test,
                ScikitModel, weight_decay, fit_intercept, max_iter, frac_to_remove,
                use_slab, use_loss,
                total_epsilon, epsilon_pos, epsilon_neg,
                theta_idx, obj,
                all_results, best_attacks)

    if timed:
        print('Saving timing...')
        timed_results_path = datasets.get_timed_results_npz_path(
            dataset_name,
            weight_decay,
            percentile,
            attack_label)
        datasets.safe_makedirs(timed_results_path)
        np.savez(
            timed_results_path,
            times_taken=times_taken,
            Xs_poison=Xs_poison,
            Ys_poison=Ys_poison,
            nums_copies=nums_copies)

    if not timed:
        print('\n#### Results ####\n')
        idx_train = slice(0, X_train.shape[0])
        for epsilon in epsilons:
            print('For epsilon %s:' % epsilon)
            print('  Test accuracy: %s' % best_attacks[epsilon]['best_acc'])
            print('  Theta idx: %s' % best_attacks[epsilon]['theta_idx'])
            print('  Epsilon pos: %s' % best_attacks[epsilon]['epsilon_pos'])
            print('  Epsilon neg: %s' % best_attacks[epsilon]['epsilon_neg'])
            print('  Obj        : %s' % best_attacks[epsilon]['obj'])
            print('')
            attack_save_path = datasets.get_target_attack_npz_path(
                dataset_name,
                epsilon,
                weight_decay,
                percentile,
                attack_label)
            datasets.safe_makedirs(attack_save_path)

            idx_poison = slice(X_train.shape[0], best_attacks[epsilon]['X_modified_round'].shape[0])

            np.savez(
                attack_save_path,
                X_poison=best_attacks[epsilon]['X_modified_round'][idx_poison, :],
                Y_poison=best_attacks[epsilon]['Y_modified'][idx_poison],
                )

        results_save_path = datasets.get_attack_results_json_path(
            dataset_name,
            weight_decay,
            percentile,
            attack_label)
        datasets.safe_makedirs(results_save_path)
        np.savez(
            results_save_path,
            best_attacks=best_attacks,
            all_results=all_results,
            thetas=f['thetas'],
            biases=f['biases']
            )

    # Just look at one theta if we are debugging
    if debug:
        break
