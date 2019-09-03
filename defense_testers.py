from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
from collections import defaultdict

import numpy as np

from sklearn import linear_model, preprocessing, cluster, metrics, svm, model_selection

import data_utils as data
import defenses
import datasets

def process_defense(
    datadef, Q, all_dists, model, weight_decay, results, use_emp, use_emp_label, defense_label,
    max_frac_to_remove, frac_increment, num_folds,
    norm=2,
    subtract_from_l2=False,
    P_datadef=None,
    dists=None):

    if dists is not None:
        assert Q is None
        assert P_datadef is None
        assert subtract_from_l2 == False

    if P_datadef is None:
        P_datadef = datadef

    if use_emp_label is not None:
        defense_emp_label = '%s_%s' % (defense_label, use_emp_label)
    else:
        defense_emp_label = defense_label

    if dists is None:
        dists = P_datadef.compute_dists_under_Q_over_dataset(
            Q=Q,
            use_emp_centroids=use_emp,
            subtract_from_l2=subtract_from_l2,
            norm=norm).reshape(-1, 1)

    results['dist_labels'].append(defense_emp_label)
    all_dists = np.concatenate((all_dists, dists), axis=1)
    auto_threshold_and_retrain(
        datadef, dists, model, weight_decay, results, defense_emp_label,
        max_frac_to_remove=max_frac_to_remove,
        frac_increment=frac_increment,
        num_folds=num_folds)

    return all_dists, results


def auto_threshold_and_retrain(
    datadef,
    dists,
    model,
    weight_decay,
    results,
    defense_emp_label,
    max_frac_to_remove=0.30,
    frac_increment=0.05,
    num_folds=5):

    def perc_format(frac):
        return "{:6.2f}".format(frac * 100)

    # We always remove at least frac_increment of the data
    fracs_to_remove = np.linspace(
        frac_increment, # 0,
        max_frac_to_remove,
        int(np.round(max_frac_to_remove / frac_increment))) # used to have a +1

    train_accs = np.zeros(len(fracs_to_remove))
    val_accs = np.zeros(len(fracs_to_remove))
    test_accs = np.zeros(len(fracs_to_remove))
    fracs_of_good_points_kept = np.zeros(len(fracs_to_remove))
    fracs_of_bad_points_kept = np.zeros(len(fracs_to_remove))
    train_losses_overall = np.zeros(len(fracs_to_remove))
    train_losses_clean = np.zeros(len(fracs_to_remove))
    train_losses_poison = np.zeros(len(fracs_to_remove))
    test_losses = np.zeros(len(fracs_to_remove))
    nums_removed_by_class = []
    nums_bad_points_by_class = []
    nums_bad_points_removed_by_class = []

    best_val_acc = -1
    for idx, frac_to_remove in enumerate(fracs_to_remove):

        if frac_to_remove == 0: # Use entire modified data
            assert NotImplementedError #need to handle retrain_results
            train_acc = results['test']['modified']['train_acc_overall']
            val_acc = results['cv_val_acc_modified']
            test_acc = results['test']['modified']['test_acc']
            frac_of_good_points_kept = 1.0
            frac_of_bad_points_kept = 1.0
        else:
            retrain_results = datadef.remove_and_retrain(
                dists,
                model,
                weight_decay,
                frac_to_remove,
                num_folds=num_folds)

        if num_folds:
            print('    Removing %s%%:  Train acc: %.3f     Validation acc: %.3f     Test acc: %.3f     %% good data kept: %s     %% bad data kept: %s' % (
                perc_format(frac_to_remove),
                retrain_results['train_acc'],
                retrain_results['val_acc'],
                retrain_results['test_acc'],
                perc_format(retrain_results['frac_of_good_points_kept']),
                perc_format(retrain_results['frac_of_bad_points_kept'])))
        else:
            print('    Removing %s%%:  Train acc: %.3f     Test acc: %.3f     %% good data kept: %s     %% bad data kept: %s' % (
                perc_format(frac_to_remove),
                retrain_results['train_acc'],
                retrain_results['test_acc'],
                perc_format(retrain_results['frac_of_good_points_kept']),
                perc_format(retrain_results['frac_of_bad_points_kept'])))

        train_accs[idx] = retrain_results['train_acc']
        val_accs[idx] = retrain_results['val_acc']
        test_accs[idx] = retrain_results['test_acc']
        train_losses_overall[idx] = retrain_results['train_loss_overall']
        train_losses_clean[idx] = retrain_results['train_loss_clean']
        train_losses_poison[idx] = retrain_results['train_loss_poison']
        test_losses[idx] = retrain_results['test_loss']
        fracs_of_good_points_kept[idx] = retrain_results['frac_of_good_points_kept']
        fracs_of_bad_points_kept[idx] = retrain_results['frac_of_bad_points_kept']
        nums_removed_by_class.append(retrain_results['num_removed_by_class'])
        nums_bad_points_by_class.append(retrain_results['num_bad_points_by_class'])
        nums_bad_points_removed_by_class.append(retrain_results['num_bad_points_removed_by_class'])
        if num_folds:
            if best_val_acc < retrain_results['val_acc']:
                best_val_acc = retrain_results['val_acc']
                best_idx = idx

    if num_folds:
        diff_in_test_acc = results['test']['clean']['test_acc'] - test_accs[best_idx]
        leverage = diff_in_test_acc / datadef.epsilon

        print('    Defense auto-selected removing %s of the data.' % fracs_to_remove[best_idx])
        print('    Best test acc: %.3f' % test_accs[best_idx])
        print('    Leverage: %.3f' % leverage)

    results[defense_emp_label] = defaultdict(dict)
    results[defense_emp_label]['fracs_to_remove'] = fracs_to_remove
    results[defense_emp_label]['train_accs'] = train_accs
    results[defense_emp_label]['val_accs'] = val_accs
    results[defense_emp_label]['test_accs'] = test_accs
    results[defense_emp_label]['train_losses_overall'] = train_losses_overall
    results[defense_emp_label]['train_losses_clean'] = train_losses_clean
    results[defense_emp_label]['train_losses_poison'] = train_losses_poison
    results[defense_emp_label]['test_losses'] = test_losses
    results[defense_emp_label]['fracs_of_good_points_kept'] = fracs_of_good_points_kept
    results[defense_emp_label]['fracs_of_bad_points_kept'] = fracs_of_bad_points_kept
    results[defense_emp_label]['nums_removed_by_class'] = nums_removed_by_class
    results[defense_emp_label]['nums_bad_points_by_class'] = nums_bad_points_by_class
    results[defense_emp_label]['nums_bad_points_removed_by_class'] = nums_bad_points_removed_by_class
    if num_folds:
        results[defense_emp_label]['cv_frac_to_remove'] = fracs_to_remove[best_idx]
        results[defense_emp_label]['cv_test_acc'] = test_accs[best_idx]
        results[defense_emp_label]['leverage'] = leverage

    return results
