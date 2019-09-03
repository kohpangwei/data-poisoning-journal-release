from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import data_utils as data
import defenses

def rround_if_needed(X_modified, Y_modified,
                     X_train, Y_train,
                     dataset_name,
                     no_round,
                     repeat_points):

    # Only round if we need to!
    if (dataset_name in ['enron', 'imdb']) and (no_round == False):
        X_poison = X_modified[X_train.shape[0]:, :]
        Y_poison = Y_modified[X_train.shape[0]:]
        X_modified_round = data.vstack(
            X_train,
            data.rround_with_repeats(X_poison, Y_poison, repeat_points))
    else:
        X_modified_round = X_modified

    return X_modified_round


def test_def_and_update_results(X_modified_round, Y_modified,
                                X_train, Y_train,
                                X_test, Y_test,
                                ScikitModel, weight_decay, fit_intercept, max_iter, frac_to_remove,
                                use_slab, use_loss,
                                total_epsilon, epsilon_pos, epsilon_neg,
                                theta_idx, obj,
                                all_results, best_attacks):

    idx_train = slice(0, X_train.shape[0])
    idx_poison = slice(X_train.shape[0], X_modified_round.shape[0])

    datadef = defenses.DataDef(X_modified_round, Y_modified, X_test, Y_test, idx_train, idx_poison)

    test_acc_before_defense, test_acc_after_defense = datadef.eval_model(
        ScikitModel, weight_decay, fit_intercept, max_iter, frac_to_remove,
        use_slab=use_slab,
        use_loss=use_loss,
        verbose=True)

    all_results[total_epsilon]['test acc after rounding, before defense'].append(test_acc_before_defense)
    all_results[total_epsilon]['test acc after rounding, after defense'].append(test_acc_after_defense)
    all_results[total_epsilon]['theta_idx'].append(theta_idx)
    all_results[total_epsilon]['theta_distance'].append(obj)
    all_results[total_epsilon]['epsilon_pos'].append(epsilon_pos)
    all_results[total_epsilon]['epsilon_neg'].append(epsilon_neg)

    if (best_attacks[total_epsilon]['best_acc'] is None) or (test_acc_after_defense < best_attacks[total_epsilon]['best_acc']):
        best_attacks[total_epsilon]['best_acc'] = test_acc_after_defense
        best_attacks[total_epsilon]['theta_idx'] = theta_idx
        best_attacks[total_epsilon]['epsilon_pos'] = epsilon_pos
        best_attacks[total_epsilon]['epsilon_neg'] = epsilon_neg
        best_attacks[total_epsilon]['X_modified_round'] = X_modified_round
        best_attacks[total_epsilon]['Y_modified'] = Y_modified
        best_attacks[total_epsilon]['obj'] = obj

    return all_results, best_attacks
