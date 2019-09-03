from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import json
import numpy as np
import scipy.sparse as sparse

import defenses
import upper_bounds

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            assert len(np.shape(obj)) == 1 # Can only handle 1D ndarrays
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.int16):
            return str(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

def get_class_map():
    return {-1: 0, 1: 1}

def get_centroids(X, Y, class_map):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    for y in set(Y):
        centroids[class_map[y], :] = np.mean(X[Y == y, :], axis=0)
    return centroids

def get_centroid_vec(centroids):
    assert centroids.shape[0] == 2
    centroid_vec = centroids[0, :] - centroids[1, :]
    centroid_vec /= np.linalg.norm(centroid_vec)
    centroid_vec = np.reshape(centroid_vec, (1, -1))
    return centroid_vec

# Can speed this up if necessary
def get_sqrt_inv_cov(X, Y, class_map):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    sqrt_inv_covs = np.zeros((num_classes, num_features, num_features))

    for y in set(Y):
        cov = np.cov(X[Y == y, :], rowvar=False)
        U_cov, S_cov, _ = np.linalg.svd(cov + 1e-6 * np.eye(num_features))
        print('    min eigenvalue of cov after 1e-6 reg is %s' % np.min(S_cov))
        sqrt_inv_covs[class_map[y], ...] = U_cov.dot(np.diag(1 / np.sqrt(S_cov)).dot(U_cov.T))

    return sqrt_inv_covs

# Can speed this up if necessary
def get_data_params(X, Y, percentile):
    num_classes = len(set(Y))
    num_features = X.shape[1]
    centroids = np.zeros((num_classes, num_features))
    class_map = get_class_map()
    centroids = get_centroids(X, Y, class_map)

    # Get radii for sphere
    sphere_radii = np.zeros(2)
    dists = defenses.compute_dists_under_Q(
        X, Y,
        Q=None,
        centroids=centroids,
        class_map=class_map,
        norm=2)
    for y in set(Y):
        sphere_radii[class_map[y]] = np.percentile(dists[Y == y], percentile)

    # Get vector between centroids
    centroid_vec = get_centroid_vec(centroids)

    # Get radii for slab
    slab_radii = np.zeros(2)
    for y in set(Y):
        dists = np.abs(
            (X[Y == y, :].dot(centroid_vec.T) - centroids[class_map[y], :].dot(centroid_vec.T)))
        slab_radii[class_map[y]] = np.percentile(dists, percentile)

    return class_map, centroids, centroid_vec, sphere_radii, slab_radii


def vstack(A, B):
    if (sparse.issparse(A) or sparse.issparse(B)):
        return sparse.vstack((A, B), format='csr')
    else:
        return np.concatenate((A, B), axis=0)


def add_points(x, y, X, Y, num_copies=1):
    if num_copies == 0:
        return X, Y
    x = np.array(x).reshape(-1)

    if sparse.issparse(X):
        X_modified = sparse.vstack((
            X,
            sparse.csr_matrix(
                np.tile(x, num_copies).reshape(-1, len(x)))))
    else:
        X_modified = np.append(
            X,
            np.tile(x, num_copies).reshape(-1, len(x)),
            axis=0)
    Y_modified = np.append(Y, np.tile(y, num_copies))
    return X_modified, Y_modified


def copy_random_points(X, Y, mask_to_choose_from=None, target_class=1, num_copies=1,
                       random_seed=18, replace=False):
    # Only copy from points where mask_to_choose_from == True

    np.random.seed(random_seed)
    combined_mask = (np.array(Y, dtype=int) == target_class)
    if mask_to_choose_from is not None:
        combined_mask = combined_mask & mask_to_choose_from

    idx_to_copy = np.random.choice(
        np.where(combined_mask)[0],
        size=num_copies,
        replace=replace)

    if sparse.issparse(X):
        X_modified = sparse.vstack((X, X[idx_to_copy, :]))
    else:
        X_modified = np.append(X, X[idx_to_copy, :], axis=0)
    Y_modified = np.append(Y, Y[idx_to_copy])
    return X_modified, Y_modified


def threshold(X):
    return np.clip(X, 0, np.max(X))


def rround(X, random_seed=3, return_sparse=True):
    if sparse.issparse(X):
        X = X.toarray()

    X_frac, X_int = np.modf(X)
    X_round = X_int + (np.random.random_sample(X.shape) < X_frac)
    if return_sparse:
        return sparse.csr_matrix(X_round)
    else:
        return X_round


def rround_with_repeats(X, Y, repeat_points, random_seed=3, return_sparse=True):

    X_round = rround(X, random_seed=random_seed, return_sparse=return_sparse)

    assert Y.shape[0] == X.shape[0]

    if repeat_points > 1:
        pos_idx = 0
        neg_idx = 0
        for i in range(X_round.shape[0]):
            if Y[i] == 1:
                if pos_idx % repeat_points == 0:
                    last_pos_x = X_round[i, :]
                else:
                    X_round[i, :] = last_pos_x
                pos_idx += 1
            else:
                if neg_idx % repeat_points == 0:
                    last_neg_x = X_round[i, :]
                else:
                    X_round[i, :] = last_neg_x
                neg_idx += 1

    return X_round


def project_onto_sphere(X, Y, radii, centroids, class_map):

    for y in set(Y):
        idx = class_map[y]
        radius = radii[idx]
        centroid = centroids[idx, :]

        shifts_from_center = X[Y == y, :] - centroid
        dists_from_center = np.linalg.norm(shifts_from_center, axis=1)

        shifts_from_center[dists_from_center > radius, :] *= radius / np.reshape(dists_from_center[dists_from_center > radius], (-1, 1))
        X[Y == y, :] = shifts_from_center + centroid

        print("Number of (%s) points projected onto sphere: %s" % (y, np.sum(dists_from_center > radius)))

    return X


def project_onto_slab(X, Y, v, radii, centroids, class_map):
    """
    v^T x needs to be within radius of v^T centroid.
    v is 1 x d and normalized.
    """
    v = np.reshape(v / np.linalg.norm(v), (1, -1))

    for y in set(Y):
        idx = class_map[y]
        radius = radii[idx]
        centroid = centroids[idx, :]

        # If v^T x is too large, then dists_along_v is positive
        # If it's too small, then dists_along_v is negative
        dists_along_v = (X[Y == y, :] - centroid).dot(v.T)
        shifts_along_v = np.reshape(
            dists_along_v - np.clip(dists_along_v, -radius, radius),
            (1, -1))
        X[Y == y, :] -= shifts_along_v.T.dot(v)

        print("Number of (%s) points projected onto slab: %s" % (y, np.sum(np.abs(dists_along_v) > radius)))

    return X

def get_projection_fn(
    X_clean,
    Y_clean,
    sphere=True,
    slab=True,
    non_negative=False,
    less_than_one=False,
    use_lp_rounding=False,
    percentile=90):

    goal = 'find_nearest_point'
    class_map, centroids, centroid_vec, sphere_radii, slab_radii = get_data_params(X_clean, Y_clean, percentile)
    if use_lp_rounding or non_negative or less_than_one or (sphere and slab):
        if use_lp_rounding:
            projector = upper_bounds.Minimizer(
                d=X_clean.shape[1],
                use_sphere=sphere,
                use_slab=slab,
                non_negative=non_negative,
                less_than_one=less_than_one,
                constrain_max_loss=False,
                goal=goal,
                X=X_clean
                )
            projector = upper_bounds.Minimizer(
                d=X_clean.shape[1],
                use_sphere=sphere,
                use_slab=slab,
                non_negative=non_negative,
                less_than_one=less_than_one,
                constrain_max_loss=False,
                goal=goal,
                X=X_clean
                )
        else:
            projector = upper_bounds.Minimizer(
                d=X_clean.shape[1],
                use_sphere=sphere,
                use_slab=slab,
                non_negative=non_negative,
                less_than_one=less_than_one,
                constrain_max_loss=False,
                goal=goal
                )

        # Add back low-rank projection if we move back to just sphere+slab
        def project_onto_feasible_set(
            X, Y,
            theta=None,
            bias=None,
            ):
            num_examples = X.shape[0]
            proj_X = np.zeros_like(X)

            for idx in range(num_examples):
                x = X[idx, :]
                y = Y[idx]
                class_idx = class_map[y]

                centroid = centroids[class_idx, :]
                sphere_radius = sphere_radii[class_idx]
                slab_radius = slab_radii[class_idx]
                proj_X[idx, :] = projector.minimize_over_feasible_set(
                    None,
                    x,
                    centroid,
                    centroid_vec,
                    sphere_radius,
                    slab_radius)

            num_projected = np.sum(np.max(X - proj_X, axis=1) > 1e-6)
            print('Projected %s examples.' % num_projected)
            return proj_X

    else:
        def project_onto_feasible_set(X, Y, theta=None, bias=None):
            if sphere:
                X = project_onto_sphere(X, Y, sphere_radii, centroids, class_map)

            elif slab:
                X = project_onto_slab(X, Y, centroid_vec, slab_radii, centroids, class_map)
            return X

    return project_onto_feasible_set


def filter_points_outside_feasible_set(X, Y,
                                       centroids, centroid_vec,
                                       sphere_radii, slab_radii,
                                       class_map):
    sphere_dists = defenses.compute_dists_under_Q(
        X,
        Y,
        Q=None,
        centroids=centroids,
        class_map=class_map)
    slab_dists = defenses.compute_dists_under_Q(
        X,
        Y,
        Q=centroid_vec,
        centroids=centroids,
        class_map=class_map)

    idx_to_keep = np.array([True] * X.shape[0])
    for y in set(Y):
        idx_to_keep[np.where(Y == y)[0][sphere_dists[Y == y] > sphere_radii[class_map[y]]]] = False
        idx_to_keep[np.where(Y == y)[0][slab_dists[Y == y] > slab_radii[class_map[y]]]] = False

    print(np.sum(idx_to_keep))
    return X[idx_to_keep, :], Y[idx_to_keep]
