from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import argparse
import time

import numpy as np

import scipy.sparse as sparse
import data_utils as data
import datasets
import upper_bounds
import defenses
import iterative_attack
from upper_bounds import hinge_loss, hinge_grad, logistic_grad
from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base

import tensorflow as tf

def get_projection_fn_for_dataset(dataset_name, X, Y, use_slab, use_LP, percentile):
    if dataset_name in ['enron', 'imdb']:
        projection_fn = data.get_projection_fn(
            X, Y,
            sphere=True,
            slab=use_slab,
            non_negative=True,
            less_than_one=False,
            use_lp_rounding=use_LP,
            percentile=percentile)
    elif dataset_name in ['mnist_17']:
        projection_fn = data.get_projection_fn(
            X, Y,
            sphere=True,
            slab=use_slab,
            non_negative=True,
            less_than_one=True,
            use_lp_rounding=False,
            percentile=percentile)
    elif dataset_name in ['dogfish']:
        projection_fn = data.get_projection_fn(
            X, Y,
            sphere=True,
            slab=use_slab,
            non_negative=False,
            less_than_one=False,
            use_lp_rounding=False,
            percentile=percentile)
    return projection_fn

np.random.seed(1)

fit_intercept = True
initial_learning_rate = 0.001
keep_probs = None
decay_epochs = [1000, 10000]
num_classes = 2
batch_size = 100
temp = 0.001
use_copy = True
use_LP = True
loss_type = 'normal_loss'

parser = argparse.ArgumentParser()
parser.add_argument('--em_iter', default=1)
parser.add_argument('--total_grad_iter', default=300)
parser.add_argument('--use_slab', action='store_true')
parser.add_argument('--dataset', default='enron')
parser.add_argument('--percentile', default=90)
parser.add_argument('--epsilon', default=0.03)
parser.add_argument('--step_size', default=0.1)
parser.add_argument('--use_train', action="store_true")
parser.add_argument('--baseline', action="store_true") # means no LP, no copy, and no smooth
parser.add_argument('--baseline_smooth', action="store_true") # means no LP, no copy
parser.add_argument('--no_LP', action="store_true")
parser.add_argument('--timed', action="store_true")
args = parser.parse_args()

dataset_name = args.dataset
use_slab = args.use_slab
epsilon = float(args.epsilon)
step_size = float(args.step_size)
percentile = int(np.round(float(args.percentile)))
max_em_iter = int(np.round(float(args.em_iter)))
total_grad_iter = int(np.round(float(args.total_grad_iter)))
use_train = args.use_train
baseline = args.baseline
baseline_smooth = args.baseline_smooth
no_LP = args.no_LP
timed = args.timed

output_root = os.path.join(datasets.OUTPUT_FOLDER, dataset_name, 'influence_data')
datasets.safe_makedirs(output_root)

print('epsilon: %s' % epsilon)
print('use_slab: %s' % use_slab)

if dataset_name == 'enron':
    weight_decay = 0.09
elif dataset_name == 'mnist_17':
    weight_decay = 0.01
elif dataset_name == 'dogfish':
    weight_decay = 1.1

if baseline:
    temp = 0
    assert dataset_name == 'enron'
    assert not baseline_smooth
    assert not use_train
    use_copy = False
    use_LP = False
    percentile = 80

if baseline_smooth:
    assert dataset_name == 'enron'
    assert not baseline
    assert not use_train
    use_copy = False
    use_LP = False
    percentile = 80

if no_LP:
    assert dataset_name == 'enron'
    use_LP = False
    percentile = 80

model_name = 'smooth_hinge_%s_sphere-True_slab-%s_start-copy_lflip-True_step-%s_t-%s_eps-%s_wd-%s_rs-1' % (
                dataset_name, use_slab,
                step_size, temp, epsilon, weight_decay)
if percentile != 90:
    model_name = model_name + '_percentile-%s' % percentile
model_name += '_em-%s' % max_em_iter
if baseline:
    model_name = model_name + '_baseline'
if baseline_smooth:
    model_name = model_name + '_baseline-smooth'
if no_LP:
    model_name = model_name + '_no-LP'
if timed:
    model_name = model_name + '_timed'

if max_em_iter == 0:
    num_grad_iter_per_em = total_grad_iter
else:
    assert total_grad_iter % max_em_iter == 0
    num_grad_iter_per_em = int(np.round(total_grad_iter / max_em_iter))

X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

if sparse.issparse(X_train):
    X_train = X_train.toarray()
if sparse.issparse(X_test):
    X_test = X_test.toarray()

if use_train:
    X_test = X_train
    Y_test = Y_train

class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
    X_train, Y_train, percentile=percentile)

feasible_flipped_mask = iterative_attack.get_feasible_flipped_mask(
    X_train, Y_train,
    centroids,
    centroid_vec,
    sphere_radii,
    slab_radii,
    class_map,
    use_slab=use_slab)

X_modified, Y_modified, indices_to_poison, copy_array = iterative_attack.init_gradient_attack_from_mask(
    X_train, Y_train,
    epsilon,
    feasible_flipped_mask,
    use_copy=use_copy)

tf.reset_default_graph()

input_dim = X_train.shape[1]
train = DataSet(X_train, Y_train)
validation = None
test = DataSet(X_test, Y_test)
data_sets = base.Datasets(train=train, validation=validation, test=test)

model = SmoothHinge(
    input_dim=input_dim,
    temp=temp,
    weight_decay=weight_decay,
    use_bias=True,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    decay_epochs=None,
    mini_batch=False,
    train_dir=output_root,
    log_dir='log',
    model_name=model_name)

model.update_train_x_y(X_modified, Y_modified)
model.train()

if timed:
    start_time = time.time()
else:
    start_time = None

num_em_iters = max(max_em_iter, 1)

for em_iter in range(num_em_iters):

    print('\n\n##### EM iter %s #####' % em_iter)
    X_modified = model.data_sets.train.x
    Y_modified = model.data_sets.train.labels

    if max_em_iter == 0:
        projection_fn = get_projection_fn_for_dataset(
            dataset_name,
            X_train,
            Y_train,
            use_slab,
            use_LP,
            percentile)
    else:
        projection_fn = get_projection_fn_for_dataset(
            dataset_name,
            X_modified,
            Y_modified,
            use_slab,
            use_LP,
            percentile)

    iterative_attack.iterative_attack(
        model,
        indices_to_poison=indices_to_poison,
        test_idx=None,
        test_description=None,
        step_size=step_size,
        num_iter=num_grad_iter_per_em,
        loss_type=loss_type,
        projection_fn=projection_fn,
        output_root=output_root,
        num_copies=copy_array,
        stop_after=2,
        start_time=start_time)
