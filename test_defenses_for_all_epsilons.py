from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import argparse
import json
import shutil
from collections import defaultdict
from subprocess import call

import numpy as np

import datasets

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='One of: imdb, enron, dogfish')
parser.add_argument('attack_label', help='One of: standard')
parser.add_argument('--percentile', default=90)
parser.add_argument('--epsilon')
parser.add_argument('--run_in_sequence', action="store_true")
parser.add_argument("--debug", help="Changes parameters so that everything is sped up.", action="store_true")

args = parser.parse_args()

dataset_name = args.dataset_name
attack_label = args.attack_label
percentile = args.percentile
target_epsilon = args.epsilon
debug = args.debug
run_in_sequence = args.run_in_sequence

if dataset_name == 'all':
    dataset_names = ['enron', 'mnist_17', 'dogfish', 'imdb']
else:
    dataset_names = [dataset_name]

if attack_label == 'all':
    attack_labels = [] # fill up #
else:
    attack_labels = [attack_label]

for dataset_name in dataset_names:
    assert dataset_name in ['enron', 'mnist_17', 'dogfish', 'imdb']
    if dataset_name == 'enron':
        weight_decay = 0.09
    elif dataset_name == 'mnist_17':
        weight_decay = 0.01
    elif dataset_name == 'dogfish':
        weight_decay = 1.1
    elif dataset_name == 'imdb':
        weight_decay = 0.01

    attack_folder = datasets.get_target_attack_folder(dataset_name)

    for attack_label in attack_labels:
        if attack_label == 'mm-nips':
            label_string = "label-%s.mat" % attack_label
        else:
            label_string = "label-%s.npz" % attack_label

        percentile_string = "percentile-%s" % percentile
        max_frac_to_remove = 0.05
        for filename in os.listdir(attack_folder):
            if (filename.endswith(label_string) and (percentile_string in filename)):

                eps_string = filename.split('epsilon-')[1].split('_')[0]
                eps = float(eps_string)

                if (target_epsilon is not None) and (eps_string != target_epsilon):
                    print('eps %s is not target %s, so skipping' % (eps_string, target_epsilon))
                    continue

                if eps > 0.03001:
                    print('eps is %s, so skipping' % eps)
                    continue
                print(filename)

                command = 'python test_defenses.py %s %s --weight_decay %s --max_frac_to_remove %s ' % (
                    dataset_name,
                    filename,
                    weight_decay,
                    max_frac_to_remove)
                if debug:
                    command += ' --debug'
                if target_epsilon is None and run_in_sequence == False:
                    command += ' &'
                print(command)
                call(command, shell=True)
