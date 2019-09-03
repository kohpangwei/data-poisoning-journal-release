from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import copy
import sys

import numpy as np
from sklearn import metrics, model_selection, neighbors

import scipy.linalg as slin
import scipy.sparse as sparse

import upper_bounds
import data_utils as data


def remove_quantile(X, Y, dists, frac_to_remove):
    """
    Removes the frac_to_remove points from X and Y with the highest value in dists.
    This works separately for each class.
    """
    if len(dists.shape) == 2: # Accept column vectors but reshape
        assert dists.shape[1] == 1
        dists = np.reshape(dists, -1)

    assert len(dists.shape) == 1
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == len(dists)
    assert 0 <= frac_to_remove
    assert frac_to_remove <= 1

    frac_to_keep = 1.0 - frac_to_remove
    num_removed_by_class = {}

    idx_to_keep = []
    for y in set(Y):
        num_to_keep = int(np.round(frac_to_keep * np.sum(Y == y)))
        num_removed_by_class[str(y)] = int(np.round(np.sum(Y == y))) - num_to_keep

        idx_to_keep.append(
            np.where(Y == y)[0][np.argsort(dists[Y == y])[:num_to_keep]])

    idx_to_keep = np.concatenate(idx_to_keep)

    X_def = X[idx_to_keep, :]
    Y_def = Y[idx_to_keep]

    return X_def, Y_def, idx_to_keep, num_removed_by_class


def compute_dists_under_Q(
    X, Y,
    Q,
    subtract_from_l2=False, #If this is true, computes ||x - mu|| - ||Q(x - mu)||
    centroids=None,
    class_map=None,
    norm=2):
    """
    Computes ||Q(x - mu)|| in the corresponding norm.
    Returns a vector of length num_examples (X.shape[0]).
    If centroids is not specified, calculate it from the data.
    If Q has dimension 3, then each class gets its own Q.
    """
    if (centroids is not None) or (class_map is not None):
        assert (centroids is not None) and (class_map is not None)
    if subtract_from_l2:
        assert Q is not None
    if Q is not None and len(Q.shape) == 3:
        assert class_map is not None
        assert Q.shape[0] == len(class_map)

    if norm == 1:
        metric = 'manhattan'
    elif norm == 2:
        metric = 'euclidean'
    else:
        raise ValueError('norm must be 1 or 2')

    Q_dists = np.zeros(X.shape[0])
    if subtract_from_l2:
        L2_dists = np.zeros(X.shape[0])

    for y in set(Y):
        if centroids is not None:
            mu = centroids[class_map[y], :]
        else:
            mu = np.mean(X[Y == y, :], axis=0)
        mu = mu.reshape(1, -1)

        if Q is None:   # assume Q = identity
            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                X[Y == y, :],
                mu,
                metric=metric).reshape(-1)

        else:
            if len(Q.shape) == 3:
                current_Q = Q[class_map[y], ...]
            else:
                current_Q = Q

            if sparse.issparse(X):
                XQ = X[Y == y, :].dot(current_Q.T)
            else:
                XQ = current_Q.dot(X[Y == y, :].T).T
            muQ = current_Q.dot(mu.T).T

            Q_dists[Y == y] = metrics.pairwise.pairwise_distances(
                XQ,
                muQ,
                metric=metric).reshape(-1)

            if subtract_from_l2:
                L2_dists[Y == y] = metrics.pairwise.pairwise_distances(
                    X[Y == y, :],
                    mu,
                    metric=metric).reshape(-1)
                Q_dists[Y == y] = np.sqrt(np.square(L2_dists[Y == y]) - np.square(Q_dists[Y == y]))

    return Q_dists


def find_feasible_label_flips_in_sphere(X, Y, percentile):
    class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
        X,
        Y,
        percentile=percentile)

    sphere_dists_flip = compute_dists_under_Q(
        X, -Y,
        Q=None,
        subtract_from_l2=False,
        centroids=centroids,
        class_map=class_map,
        norm=2)

    feasible_flipped_mask = np.zeros(X.shape[0], dtype=bool)

    for y in set(Y):
        class_idx_flip = class_map[-y]
        sphere_radius_flip = sphere_radii[class_idx_flip]

        feasible_flipped_mask[Y == y] = (sphere_dists_flip[Y == y] <= sphere_radius_flip)

    return feasible_flipped_mask


class DataDef(object):
    def __init__(self, X_modified, Y_modified, X_test, Y_test, idx_train, idx_poison):
        self.X_modified = X_modified
        self.Y_modified = Y_modified
        self.X_test = X_test
        self.Y_test = Y_test
        self.idx_train = idx_train
        self.idx_poison = idx_poison

        self.X_train = X_modified[idx_train, :]
        self.Y_train = Y_modified[idx_train]
        self.X_poison = X_modified[idx_poison, :]
        self.Y_poison = Y_modified[idx_poison]

        self.class_map = data.get_class_map()
        self.emp_centroids = data.get_centroids(self.X_modified, self.Y_modified, self.class_map)
        self.true_centroids = data.get_centroids(self.X_train, self.Y_train, self.class_map)
        self.emp_centroid_vec = data.get_centroid_vec(self.emp_centroids)
        self.true_centroid_vec = data.get_centroid_vec(self.true_centroids)

        # Fraction of bad data / good data (so in total, there's 1+epsilon * good data )
        self.epsilon = self.X_poison.shape[0] / self.X_train.shape[0]

    def compute_dists_under_Q_over_dataset(
        self,
        Q,
        subtract_from_l2=False, #If this is true, plots ||x - mu|| - ||Q(x - mu)||
        use_emp_centroids=False,
        norm=2):

        if use_emp_centroids:
            centroids = self.emp_centroids
        else:
            centroids = self.true_centroids

        dists = compute_dists_under_Q(
            self.X_modified, self.Y_modified,
            Q,
            subtract_from_l2=subtract_from_l2,
            centroids=centroids,
            class_map=self.class_map,
            norm=norm)

        return dists

    def get_losses(self, w, b):
        # This removes the max term from the hinge, so you can get negative loss if it's fit well
        losses = 1 - self.Y_modified * (self.X_modified.dot(w) + b)
        return losses

    def get_sqrt_inv_covs(self, use_emp=False):
        if use_emp:
            sqrt_inv_covs = data.get_sqrt_inv_cov(self.X_modified, self.Y_modified, self.class_map)
        else:
            sqrt_inv_covs = data.get_sqrt_inv_cov(self.X_train, self.Y_train, self.class_map)
        return sqrt_inv_covs

    def get_knn_dists(self, num_neighbors, use_emp=False):
        metric = 'euclidean'
        if use_emp:
            nbrs = neighbors.NearestNeighbors(
                n_neighbors=num_neighbors,
                metric=metric).fit(
                    self.X_modified)
        else:
            nbrs = neighbors.NearestNeighbors(
                n_neighbors=num_neighbors,
                metric=metric).fit(
                    self.X_train)
        # Regardless of whether you use emp, we still want distances to the whole (modified) dataset.
        dists_to_each_neighbor, _ = nbrs.kneighbors(self.X_modified)
        return np.sum(dists_to_each_neighbor, axis=1)


    # Might be able to speed up; is svds actually performant on dense matrices?
    def project_to_low_rank(
        self,
        k,
        use_emp=False,
        get_projected_data=False):
        """
        Projects to the rank (k+2) subspace defined by the top k SVs, mu_pos, and mu_neg.

        If k is None, it tries to find a good k by taking the top 1000 SVs and seeing if we can
        find some k such that sigma_k / sigma_1 < 0.1. If we can, we take the smallest such k.
        If not, we take k = 1000 or d-1. (but when we add 2 back, this seems bad?)

        Square root of the sum of squares is Frobenius norm.
        """
        if use_emp:
            X = self.X_modified
            Y = self.Y_modified
        else:
            X = self.X_train
            Y = self.Y_train

        if sparse.issparse(X):
            sq_fro_norm = sparse.linalg.norm(X, 'fro') ** 2
        else:
            sq_fro_norm = np.linalg.norm(X, 'fro') ** 2

        if k is not None:
            assert k > 0
            assert k < self.X_train.shape[1]

            U, S, V = sparse.linalg.svds(X, k=k, which='LM')

        # If k is not specified, try to automatically find a good value
        # This is a bit confusing because svds returns eigenvalues in increasing order
        # so the meaning of k is reversed
        else:
            search_k = min(1000, X.shape[1] - 1)
            target_sv_ratio = 0.95

            U, S, V = sparse.linalg.svds(X, k=search_k, which='LM')

            # Make sure it's sorted in the order we think it is...
            sort_idx = np.argsort(S)[::-1]
            S = S[sort_idx]
            V = V[sort_idx, :]
            max_sv = np.max(S)

            assert S[0] == max_sv

            sq_sv_cumsum = np.cumsum(np.power(S, 2))
            assert np.all(sq_sv_cumsum < sq_fro_norm)

            sv_ratios = sq_sv_cumsum / sq_fro_norm

            if sv_ratios[-1] > target_sv_ratio:
                k = np.where(sv_ratios > target_sv_ratio)[0][0]
            else:
                print('  Giving up -- max ratio was %s' % np.max(sv_ratios))
                k = -1

            V = V[:k, :]
            S = S[:k]

        mu_pos = np.array(np.mean(X[Y == 1, :], axis=0)).reshape(1, -1)
        mu_neg = np.array(np.mean(X[Y == -1, :], axis=0)).reshape(1, -1)

        V_mu = np.concatenate((V, mu_pos, mu_neg), axis=0)
        P = slin.orth(V_mu.T).T

        achieved_sv_ratio = np.sum(np.power(S, 2)) / sq_fro_norm

        if get_projected_data:
            PX_modified = self.X_modified.dot(P.T)
            PX_train = self.X_train.dot(P.T)
            PX_poison = self.X_poison.dot(P.T)
            return P, achieved_sv_ratio, PX_modified, PX_train, PX_poison
        else:
            return P, achieved_sv_ratio


    def find_num_points_kept(self, idx_to_keep):
        good_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        good_mask[self.idx_train] = True
        bad_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        bad_mask[self.idx_poison] = True

        keep_mask = np.zeros(self.X_modified.shape[0], dtype=bool)
        keep_mask[idx_to_keep] = True

        frac_of_good_points_kept = np.mean(keep_mask & good_mask) / np.mean(good_mask)
        frac_of_bad_points_kept = np.mean(keep_mask & bad_mask) / np.mean(bad_mask)

        num_bad_points_removed_by_class = {}
        for y in set(self.Y_modified):
            num_bad_points_removed_by_class[str(y)] = np.sum(~keep_mask & bad_mask & (self.Y_modified == y))

        return frac_of_good_points_kept, frac_of_bad_points_kept, num_bad_points_removed_by_class


    # Because this needs to handle weight decay
    # this actually creates a copy of model and changes its C
    def remove_and_retrain(
        self,
        dists,
        model,
        weight_decay,
        frac_to_remove,
        num_folds=5):

        X_def, Y_def, idx_to_keep, num_removed_by_class = remove_quantile(
            self.X_modified,
            self.Y_modified,
            dists=dists,
            frac_to_remove=frac_to_remove)

        frac_of_good_points_kept, frac_of_bad_points_kept, num_bad_points_removed_by_class = self.find_num_points_kept(idx_to_keep)

        num_bad_points_by_class = {}
        for y in set(self.Y_poison):
            num_bad_points_by_class[str(y)] = int(np.round(np.sum(self.Y_poison == y)))

        model_def = copy.deepcopy(model)
        model_def.C = 1.0 / (X_def.shape[0] * weight_decay)

        mean_cv_score = None
        if num_folds is not None:
            k_fold = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=2)

            cv_scores = model_selection.cross_val_score(
                model_def,
                X_def, Y_def,
                cv=k_fold,
                n_jobs=np.min((num_folds, 8)))
            mean_cv_score = np.mean(cv_scores)

        model_def.fit(X_def, Y_def)
        params_def = np.reshape(model_def.coef_, -1)
        bias_def = model_def.intercept_[0]

        train_acc = model_def.score(X_def, Y_def)
        test_acc = model_def.score(self.X_test, self.Y_test)
        train_loss_overall = upper_bounds.hinge_loss(params_def, bias_def, X_def, Y_def)
        train_loss_clean = upper_bounds.hinge_loss(params_def, bias_def, self.X_train, self.Y_train)
        train_loss_poison = upper_bounds.hinge_loss(params_def, bias_def, self.X_poison, self.Y_poison)
        test_loss = upper_bounds.hinge_loss(params_def, bias_def, self.X_test, self.Y_test)

        results = {}
        results['train_acc'] = train_acc
        results['val_acc'] = mean_cv_score
        results['test_acc'] = test_acc
        results['train_loss_overall'] = train_loss_overall
        results['train_loss_clean'] = train_loss_clean
        results['train_loss_poison'] = train_loss_poison
        results['test_loss'] = test_loss
        results['frac_of_good_points_kept'] = frac_of_good_points_kept
        results['frac_of_bad_points_kept'] = frac_of_bad_points_kept
        results['num_removed_by_class'] = num_removed_by_class
        results['num_bad_points_by_class'] = num_bad_points_by_class
        results['num_bad_points_removed_by_class'] = num_bad_points_removed_by_class

        return results


    def eval_model(self, ScikitModel, weight_decay, fit_intercept, max_iter, frac_to_remove,
        intercept_scaling=1,
        use_slab=False,
        use_loss=False,
        verbose=True):
        """
        Runs sphere, slab, loss
        """

        def report_test_acc(dists, def_str):
            retrain_results = self.remove_and_retrain(
                dists,
                model_def,
                weight_decay,
                frac_to_remove,
                num_folds=None)

            test_acc = retrain_results['test_acc']

            if verbose:
                train_acc = retrain_results['train_acc']
                frac_of_good_points_kept = retrain_results['frac_of_good_points_kept']
                frac_of_bad_points_kept = retrain_results['frac_of_bad_points_kept']

                print()
                print('After defending (%s):' % def_str)
                print('Train (clean+poi): %.3f' % train_acc)
                print('Test (overall or targeted)   : %.3f' % test_acc)
                print('Good points kept : %.3f%%' % (frac_of_good_points_kept*100))
                print('Bad points kept  : %.3f%%' % (frac_of_bad_points_kept*100))

            return test_acc

        C = 1.0 / (self.X_modified.shape[0] * weight_decay)
        model_round = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            random_state=24,
            max_iter=max_iter,
            verbose=True)
        model_round.fit(self.X_modified, self.Y_modified)
        test_acc_before_defense = model_round.score(self.X_test, self.Y_test)

        print()
        print('With our attack, no defenses:')
        print('Train (clean)    : %.3f' % model_round.score(self.X_train, self.Y_train))
        print('Train (clean+poi): %.3f' % model_round.score(self.X_modified, self.Y_modified))
        print('Test (overall)   : %.3f' % test_acc_before_defense)

        model_def = ScikitModel(
            C=C,
            tol=1e-8,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            random_state=24,
            max_iter=max_iter,
            verbose=True)

        # L2 defense
        dists = self.compute_dists_under_Q_over_dataset(
            Q=None,
            use_emp_centroids=True,
            norm=2)
        highest_test_acc = report_test_acc(dists, 'L2')

        # Loss defense
        if use_loss:
            dists = self.get_losses(model_round.coef_.reshape(-1), model_round.intercept_)
            highest_test_acc = max(highest_test_acc, report_test_acc(dists, 'loss'))

        # Slab defense
        if use_slab:
            dists = self.compute_dists_under_Q_over_dataset(
                Q=self.emp_centroid_vec,
                use_emp_centroids=True,
                norm=2)
            highest_test_acc = max(highest_test_acc, report_test_acc(dists, 'slab'))

        return test_acc_before_defense, highest_test_acc
