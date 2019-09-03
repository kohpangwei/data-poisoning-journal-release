from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time

import numpy as np

from scipy.linalg import orth
import scipy.sparse as sparse
from sklearn import svm

import data_utils as data
import datasets

import cvxpy as cvx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    a = sigmoid(x)
    return a * (1 - a)

def logistic_grad(w, b, X, Y):
    margins = Y * (X.dot(w) + b)
    if sparse.issparse(X):
        SMY = -sigmoid(-margins).reshape((-1, 1)) * Y.reshape((-1, 1))
        grad_w = np.array(np.mean(
            sparse.diags(SMY.reshape(-1)).dot(X),
            axis=0)).reshape(-1)
    else:
        grad_w = np.mean(
            -sigmoid(-margins).reshape((-1, 1)) * Y.reshape((-1, 1)) * X,
            axis=0)
    grad_b = np.mean(
        -sigmoid(-margins).reshape((-1, 1)) * Y.reshape((-1, 1)),
        axis=0)
    return grad_w, grad_b

def indiv_hinge_losses(w, b, X, Y):
    return np.maximum(1 - Y * (X.dot(w) + b), 0)

def hinge_loss(w, b, X, Y, sample_weights=None):
    if sample_weights is not None:
        sample_weights = sample_weights / np.sum(sample_weights)
        return np.sum(sample_weights * (np.maximum(1 - Y * (X.dot(w) + b), 0)))
    else:
        return np.mean(np.maximum(1 - Y * (X.dot(w) + b), 0))

def hinge_grad(w, b, X, Y):
    margins = Y * (X.dot(w) + b)
    sv_indicators = margins < 0.99
    if sparse.issparse(X):
        grad_w = np.sum(
            -sparse.diags(np.reshape(Y[sv_indicators], (-1))).dot(
                X[sv_indicators, :]) , axis=0) / X.shape[0]
        grad_w = np.array(grad_w).reshape(-1)
    else:
        grad_w = np.sum(
            -np.reshape(Y[sv_indicators], (-1, 1)) * X[sv_indicators, :],
             axis=0) / X.shape[0]

    grad_b = np.sum( -np.reshape(Y[sv_indicators], (-1, 1))) / X.shape[0]

    return grad_w, grad_b


def get_max_hinge_losses(
    X,
    Y,
    target_theta,
    target_bias,
    class_map,
    percentile):

    losses_at_target = indiv_hinge_losses(
        target_theta,
        target_bias,
        X,
        Y)

    max_losses = [0, 0]
    for y in set(Y):
        max_losses[class_map[y]] = np.percentile(losses_at_target[Y == y], percentile)

    return max_losses


### Minimizer / CVX

def cvx_dot(a,b):
    return cvx.sum_entries(cvx.mul_elemwise(a, b))

def get_projection_matrix(A):
    """
    Output: projection matrix P that projects a vector onto the subspace spanned by
            the columns of A
    P is A.shape[1] x A.shape[0]
    """
    P = orth(A).T
    d = A.shape[1]
    while P.shape[0] < d:
        P = np.concatenate((P, np.random.normal(size=(1, P.shape[1]))), axis=0)
        P = orth(P.T).T

    return P

class Minimizer(object):

    def __init__(
        self,
        d,
        use_sphere=True,
        use_slab=True,
        non_negative=False,
        less_than_one=False,
        constrain_max_loss=False,
        goal=None,
        X=None):

        assert goal in ['find_nearest_point', 'maximize_test_loss']

        self.use_slab = use_slab
        self.constrain_max_loss = constrain_max_loss
        self.goal = goal

        self.use_projection = not (non_negative or less_than_one or X)
        if self.use_projection:
            eff_d = 3 + (constrain_max_loss == True)
        else:
            eff_d = d


        self.cvx_x = cvx.Variable(eff_d)
        self.cvx_w = cvx.Parameter(eff_d)
        self.cvx_centroid = cvx.Parameter(eff_d)
        self.cvx_sphere_radius = cvx.Parameter(1)

        self.cvx_x_c = self.cvx_x - self.cvx_centroid

        if goal == 'find_nearest_point':
            self.objective = cvx.Minimize(cvx.pnorm(self.cvx_x - self.cvx_w, 2) ** 2)
        elif goal == 'maximize_test_loss':
            self.cvx_y = cvx.Parameter(1)
            # No need for bias term
            self.objective = cvx.Maximize(1 - self.cvx_y * cvx_dot(self.cvx_w, self.cvx_x))

        self.constraints = []
        if use_sphere:
            if X is not None:
                if sparse.issparse(X):
                    X_max = X.max(axis=0).toarray().reshape(-1)
                else:
                    X_max = np.max(X, axis=0).reshape(-1)
                X_max[X_max < 1] = 1
                X_max[X_max > 50] = 50
                self.constraints.append(self.cvx_x <= X_max)
                kmax = int(np.ceil(np.max(X_max)))

                self.cvx_env = cvx.Variable(eff_d)
                for k in range(1, kmax+1):
                    active = k <= (X_max)
                    self.constraints.append(self.cvx_env[active] >= self.cvx_x[active] * (2*k-1) - k*(k-1))

                self.constraints.append(
                    (
                        cvx.sum_entries(self.cvx_env)
                        - 2 * cvx_dot(self.cvx_centroid, self.cvx_x)
                        + cvx.sum_squares(self.cvx_centroid)
                    )
                    < (self.cvx_sphere_radius ** 2)
                )
            else:
                self.constraints.append(cvx.pnorm(self.cvx_x_c, 2) ** 2 < self.cvx_sphere_radius ** 2)

        if use_slab:
            self.cvx_centroid_vec = cvx.Parameter(eff_d)
            self.cvx_slab_radius = cvx.Parameter(1)
            self.constraints.append(cvx_dot(self.cvx_centroid_vec, self.cvx_x_c) < self.cvx_slab_radius)
            self.constraints.append(-cvx_dot(self.cvx_centroid_vec, self.cvx_x_c) < self.cvx_slab_radius)

        if non_negative:
            self.constraints.append(self.cvx_x >= 0)

        if less_than_one:
            self.constraints.append(self.cvx_x <= 1)

        if constrain_max_loss:
            self.cvx_max_loss = cvx.Parameter(1)
            self.cvx_constraint_w = cvx.Parameter(eff_d)
            self.cvx_constraint_b = cvx.Parameter(1)
            self.constraints.append(
                1 - self.cvx_y * (
                    cvx_dot(self.cvx_constraint_w, self.cvx_x) + self.cvx_constraint_b
                ) < self.cvx_max_loss)


        self.prob = cvx.Problem(self.objective, self.constraints)

    def minimize_over_feasible_set(self, y, w,
                                   centroid, centroid_vec, sphere_radius, slab_radius,
                                   max_loss=None,
                                   constraint_w=None,
                                   constraint_b=None,
                                   verbose=False):
        """
        Includes both sphere and slab.
        Returns optimal x.
        """
        start_time = time.time()
        if self.use_projection:
            A = np.concatenate(
                (
                    w.reshape(-1, 1),
                    centroid.reshape(-1, 1),
                    centroid_vec.reshape(-1, 1)
                ),
                axis=1)

            if constraint_w is not None:
                A = np.concatenate(
                    (A, constraint_w.reshape(-1, 1)),
                    axis=1)

            P = get_projection_matrix(A)

            self.cvx_w.value = P.dot(w.reshape(-1))
            self.cvx_centroid.value = P.dot(centroid.reshape(-1))
            self.cvx_centroid_vec.value = P.dot(centroid_vec.reshape(-1))
        else:
            self.cvx_w.value = w.reshape(-1)
            self.cvx_centroid.value = centroid.reshape(-1)
            self.cvx_centroid_vec.value = centroid_vec.reshape(-1)

        if self.goal == 'maximize_test_loss':
            self.cvx_y.value = y

        self.cvx_sphere_radius.value = sphere_radius
        self.cvx_slab_radius.value = slab_radius

        if self.constrain_max_loss:
            self.cvx_max_loss.value = max_loss
            self.cvx_constraint_b.value = constraint_b
            if self.use_projection:
                self.cvx_constraint_w.value = P.dot(constraint_w.reshape(-1))
            else:
                self.cvx_constraint_w.value = constraint_w.reshape(-1)

        try:
            self.prob.solve(verbose=verbose, solver=cvx.SCS)
        except:
            raise
            print('centroid', self.cvx_centroid.value)
            print('centroid_vec', self.cvx_centroid_vec.value)
            print('w', self.cvx_w.value)
            print('sphere_radius', sphere_radius)
            print('slab_radius', slab_radius)
            if self.constrain_max_loss:
                print('constraint_w', self.cvx_constraint_w.value)
                print('constraint_b', self.cvx_constraint_b.value)

            print('Resolving verbosely')
            self.prob.solve(verbose=True, solver=cvx.SCS)
            raise

        x_opt = np.array(self.cvx_x.value).reshape(-1)

        if self.use_projection:
            return x_opt.dot(P)
        else:
            return x_opt


class TwoClassKKT(object):
    def __init__(
        self,
        d,
        dataset_name=None,
        X=None,
        use_slab=False,
        constrain_max_loss=False):

        self.use_slab = use_slab
        self.constrain_max_loss = constrain_max_loss

        self.cvx_x_pos = cvx.Variable(d)
        self.cvx_x_neg = cvx.Variable(d)
        self.cvx_epsilon_pos = cvx.Parameter(1)
        self.cvx_epsilon_neg = cvx.Parameter(1)
        self.cvx_g = cvx.Parameter(d)
        self.cvx_theta = cvx.Parameter(d)
        self.cvx_bias = cvx.Parameter(1)
        self.cvx_centroid_pos = cvx.Parameter(d)
        self.cvx_centroid_neg = cvx.Parameter(d)
        self.cvx_sphere_radius_pos = cvx.Parameter(1)
        self.cvx_sphere_radius_neg = cvx.Parameter(1)

        if use_slab:
            self.cvx_centroid_vec = cvx.Parameter(d)
            self.cvx_slab_radius_pos = cvx.Parameter(1)
            self.cvx_slab_radius_neg = cvx.Parameter(1)

        if constrain_max_loss:
            self.cvx_max_loss_pos = cvx.Parameter(1)
            self.cvx_max_loss_neg = cvx.Parameter(1)

        self.cvx_err = cvx.Variable(d)
        self.objective = cvx.Minimize(cvx.pnorm(self.cvx_err, 2))

        self.constraints = [
            self.cvx_g - self.cvx_epsilon_pos * self.cvx_x_pos + self.cvx_epsilon_neg * self.cvx_x_neg == self.cvx_err,
            cvx_dot(self.cvx_theta, self.cvx_x_pos) + self.cvx_bias < 1, # margin constraint, ideally should be 1
            -(cvx_dot(self.cvx_theta, self.cvx_x_neg) + self.cvx_bias) < 1 , # ideally should be 1
        ]

        if use_slab:
            self.constraints.append(cvx_dot(self.cvx_centroid_vec, self.cvx_x_pos - self.cvx_centroid_pos) < self.cvx_slab_radius_pos)
            self.constraints.append(-cvx_dot(self.cvx_centroid_vec, self.cvx_x_pos - self.cvx_centroid_pos) < self.cvx_slab_radius_pos)

            self.constraints.append(cvx_dot(self.cvx_centroid_vec, self.cvx_x_neg - self.cvx_centroid_neg) < self.cvx_slab_radius_neg)
            self.constraints.append(-cvx_dot(self.cvx_centroid_vec, self.cvx_x_neg - self.cvx_centroid_neg) < self.cvx_slab_radius_neg)

        if dataset_name in ['mnist_17', 'enron', 'imdb']:
            self.constraints.append(self.cvx_x_pos >= 0)
            self.constraints.append(self.cvx_x_neg >= 0)

        if constrain_max_loss:
            self.constraints.append(
                1 - (cvx_dot(self.cvx_theta, self.cvx_x_pos) + self.cvx_bias) < self.cvx_max_loss_pos)
            self.constraints.append(
                1 + (cvx_dot(self.cvx_theta, self.cvx_x_neg) + self.cvx_bias) < self.cvx_max_loss_neg)

        if dataset_name == 'mnist_17':
            self.constraints.append(self.cvx_x_pos <= 1)
            self.constraints.append(self.cvx_x_neg <= 1)

        # If we pass in X, do the LP/integer constraint
        if (X is not None) and (dataset_name in ['enron', 'imdb']):
            if sparse.issparse(X):
                X_max = np.max(X, axis=0).toarray().reshape(-1)
            else:
                X_max = np.max(X, axis=0).reshape(-1)
            X_max[X_max < 1] = 1
            X_max[X_max > 50] = 50

            self.constraints.append(self.cvx_x_pos <= X_max)
            self.constraints.append(self.cvx_x_neg <= X_max)
            kmax = int(np.ceil(np.max(X_max)))

            self.cvx_env_pos = cvx.Variable(d)
            self.cvx_env_neg = cvx.Variable(d)
            for k in range(1, kmax+1):
                active = k <= (X_max)
                self.constraints.append(self.cvx_env_pos[active] >= self.cvx_x_pos[active] * (2*k-1) - k*(k-1))
                self.constraints.append(self.cvx_env_neg[active] >= self.cvx_x_neg[active] * (2*k-1) - k*(k-1))

            self.constraints.append(
                (
                    cvx.sum_entries(self.cvx_env_pos)
                    - 2 * cvx_dot(self.cvx_centroid_pos, self.cvx_x_pos)
                    + cvx.sum_squares(self.cvx_centroid_pos)
                )
                < (self.cvx_sphere_radius_pos ** 2)
            )
            self.constraints.append(
                (
                    cvx.sum_entries(self.cvx_env_neg)
                    - 2 * cvx_dot(self.cvx_centroid_neg, self.cvx_x_neg)
                    + cvx.sum_squares(self.cvx_centroid_neg)
                )
                < (self.cvx_sphere_radius_neg ** 2)
            )
        else:
            self.constraints.append(cvx.pnorm(self.cvx_x_pos - self.cvx_centroid_pos, 2) ** 2 < self.cvx_sphere_radius_pos ** 2)
            self.constraints.append(cvx.pnorm(self.cvx_x_neg - self.cvx_centroid_neg, 2) ** 2 < self.cvx_sphere_radius_neg ** 2)


    def solve(self,
        g, theta,
        epsilon_pos, epsilon_neg,
        class_map, centroids, centroid_vec, sphere_radii, slab_radii,
        target_bias=None,
        target_bias_grad=None,
        max_losses=None,
        verbose=False):

        if target_bias is not None:
            assert target_bias_grad is not None
        if target_bias_grad is not None:
            assert target_bias is not None

        self.cvx_centroid_pos.value = centroids[class_map[1]].reshape(-1)
        self.cvx_centroid_neg.value = centroids[class_map[-1]].reshape(-1)
        self.cvx_centroid_vec.value = centroid_vec.reshape(-1)
        self.cvx_sphere_radius_pos.value = sphere_radii[class_map[1]]
        self.cvx_sphere_radius_neg.value = sphere_radii[class_map[-1]]

        self.cvx_g.value = g.reshape(-1)
        self.cvx_theta.value = theta.reshape(-1)

        if target_bias is not None:
            self.cvx_bias.value = target_bias
        else:
            self.cvx_bias.value = 0

        if self.use_slab:
            self.cvx_slab_radius_pos.value = slab_radii[class_map[1]]
            self.cvx_slab_radius_neg.value = slab_radii[class_map[-1]]

        if self.constrain_max_loss:
            self.cvx_max_loss_pos.value = max_losses[class_map[1]]
            self.cvx_max_loss_neg.value = max_losses[class_map[-1]]

        self.prob = cvx.Problem(self.objective, self.constraints)

        best_value = None

        # We want:
        # total_epsilon = epsilon_pos + epsilon_neg
        # target_bias_grad - epsilon_pos + epsilon_neg = 0
        # epsilon_neg = (total_epsilon - target_bias_grad) / 2
        # if (epsilon_neg < 0): epsilon_neg = 0
        # if (epsilon_neg > total_epsilon): epsilon_neg = total_epsilon
        # epsilon_pos = total_epsilon - epsilon_neg
        # if (epsilon_pos < 0): epsilon_pos = 0

        self.cvx_epsilon_pos.value = epsilon_pos
        self.cvx_epsilon_neg.value = epsilon_neg

        self.prob.solve(verbose=verbose, solver=cvx.GUROBI)          #max_iters=1000
        print('***** ', epsilon_pos, epsilon_neg, self.prob.value)

        best_value = self.prob.value
        best_epsilon_pos = epsilon_pos
        best_epsilon_neg = epsilon_neg
        best_x_pos = np.array(self.cvx_x_pos.value)
        best_x_neg = np.array(self.cvx_x_neg.value)

        if self.constrain_max_loss:
            print('Loss of x_pos is %s' % (1 - (theta.dot(best_x_pos) + target_bias)))
            print('Loss of x_neg is %s' % (1 + (theta.dot(best_x_neg) + target_bias)))

        return best_x_pos, best_x_neg, best_epsilon_pos, best_epsilon_neg

###
class GurobiSVM(object):
    def __init__(
        self,
        weight_decay):

        self.weight_decay = weight_decay

    # We'll need to iterate this so consider moving it up and getting dimension in init
    # We always normalize sample weights so that weight decay has the same effect
    def fit(
        self,
        X, Y,
        sample_weights=None,
        verbose=False):

        assert(X.shape[0] == Y.shape[0])
        if sample_weights is not None:
            assert(len(sample_weights.shape) == 1)
            assert(len(sample_weights) == X.shape[0])

        n = X.shape[0]
        d = X.shape[1]
        self.cvx_w = cvx.Variable(d)
        self.cvx_b = cvx.Variable(1)
        self.cvx_hinge_losses = cvx.Variable(n)

        if sample_weights is None:
            sample_weights = np.ones(n)

        total_sample_weights = np.sum(sample_weights)

        self.objective = cvx.Minimize(
            cvx.sum_entries(
                cvx.mul_elemwise(
                    sample_weights,
                    self.cvx_hinge_losses)) / total_sample_weights
            + 0.5 * self.weight_decay * cvx.sum_squares(self.cvx_w))
        self.constraints = [
            cvx.mul_elemwise(Y, X * self.cvx_w + self.cvx_b) >= 1 - self.cvx_hinge_losses,
            self.cvx_hinge_losses >= 0]

        self.prob = cvx.Problem(self.objective, self.constraints)
        self.prob.solve(verbose=verbose, solver=cvx.GUROBI)

        self.coef_ = np.array(self.cvx_w.value).reshape(-1)
        self.intercept_ = self.cvx_b.value

    def get_indiv_hinge_losses(self, X, Y):
        return indiv_hinge_losses(self.coef_, self.intercept_, X, Y)


class QFinder(object):
    def __init__(self, m, q_budget):
        self.cvx_q = cvx.Variable(m)
        self.cvx_loss_diffs = cvx.Parameter(m)
        self.objective = cvx.Minimize(cvx_dot(self.cvx_loss_diffs, self.cvx_q))
        self.constraints = [
            cvx.sum_entries(self.cvx_q) == q_budget,
            self.cvx_q >= 0]
        self.prob = cvx.Problem(self.objective, self.constraints)

    def solve(self, loss_diffs, verbose=False):
        self.cvx_loss_diffs.value = loss_diffs
        self.prob.solve(verbose=verbose, solver=cvx.GUROBI)
        q = np.array(self.cvx_q.value).reshape(-1)
        return q
