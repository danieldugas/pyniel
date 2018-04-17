import sklearn as skl
import matplotlib.pyplot as plt
import numpy as np
import treelib as tl

from sklearn.utils.validation import check_is_fitted

from numpy_tools.logic import ismin

TINY = 1e-8

class DeterministicAnnealingClustering(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers_ (np.ndarray): Cluster centroids y_i (n_clusters, n_features)
        cluster_probabs_ (np.ndarray): Assignment probability vectors p(y_i | x) for each sample
                                       (n_samples, n_clusters)
        bifurcation_tree_ (treelib.Tree): Tree object that contains information about cluster evolution during
                                          annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian", log={}):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.TMIN = 0.001
        self.EPS = 1e-2 # convergence tolerance for Lagrangian minimization
        self.MAX_ITER = 1000 # Max iterations for Lagrangian minimization
        self.ALPHA = 0.9
        self.DELTA_SCALE = 1e-2
        self.DEBUG_PLOT = False
        self.log = {key: [] for key in log}
        # Add more parameters, if necessary.
        self.cluster_probabs_ = None
        self.bifurcation_tree_ = tl.Tree()

    def _perturb(self, y):
        delta = np.random.normal(size=y.shape)
        delta = self.DELTA_SCALE * delta / (np.linalg.norm(delta, axis=-1)[...,None] + TINY)
        return y + delta

    def fit(self, X):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].
        Consider to use initialization and reseeding as in sklearn k-means for improved performance.

        Args:
            X (np.ndarray): Input array with shape (samples, n_features)

        Returns:
            self
        """
        np.random.seed(self.random_state)


        # Scale delta to the variance of the dataset
        self.DELTA_SCALE = self.DELTA_SCALE * np.sqrt(np.var(X, axis=0))

        if self.metric == "euclidian":
            def d(a,b, axis):
                return np.linalg.norm(a-b, axis=axis)
        else:
            raise NotImplementedError
        self.d_ = d

        px = 1.0/len(X) # equiprobable x
        dd = X - np.mean(X, axis=0)
        Cx = np.sum(px * np.matmul(dd[:,:,None], dd[:,None,:]), axis=0)
        l, _ = np.linalg.eig(Cx)
        T0 = T = 2*np.max(l) # is this valid with other metrics?

        self.cluster_centers_ = np.array([np.mean(X, axis=0)])
        self.bifurcation_tree_.create_node(0, 0, data={"T": T})

        last_F = -1 # init for convergence check
        while True:
            # Split clusters
            if len(self.cluster_centers_) < self.n_clusters and T > 0:
                clusters = np.copy(np.concatenate((self.cluster_centers_, self.cluster_centers_), axis = 0))
            else:
                clusters = np.copy(self.cluster_centers_)
            clusters = self._perturb(clusters)
            n_ef = len(self.cluster_centers_)
            # Optimize at T
            for i in range(self.MAX_ITER):
                # optimize pyx
                if T > 0:
                  pyx = np.exp(-self.d_(clusters[:,None,:],X[None,:,:], axis=-1)/T)# (K, n_x)
                else:
                  pyx = ismin(self.d_(clusters[:,None,:],X[None,:,:], axis=-1), axis=0).astype(float)
                Zx = np.sum(pyx, axis=0)
                pyx = pyx / (Zx + TINY)
                py = np.sum(pyx, axis=1) * px
                pxy = pyx.T * px / (py + TINY) # (n_x, K)
                # optimize y
                clusters = np.sum(pxy[:,:,None] * X[:,None,:], axis = 0)
                # convergence check
                F = np.sum(np.log(Zx + TINY))
                diff_F = np.abs(F - last_F)
                last_F = F
                # logging
                if 'distortion' in self.log:
                    D = np.sum(px * np.sum(pyx * self.d_(X[None,:,:], clusters[:,None,:],
                                                        axis = -1), axis=0), axis=-1)
                    self.log['distortion'].append(D)
                if 'temperature' in self.log:
                    self.log['temperature'].append(T)
                if 'n_effective_clusters' in self.log:
                    self.log['n_effective_clusters'].append(len(self.cluster_centers_))
                if 'cluster_centers' in self.log:
                    self.log['cluster_centers'].append(np.copy(self.cluster_centers_))
                # loop exit
                if diff_F < self.EPS:
                    break

            # Confirm split for most diverged cluster
            if len(self.cluster_centers_) < self.n_clusters and T > 0:
                old, new = np.split(clusters, 2)
                # New clusters should have diverged from their parent but also from other old clusters
                divergence = np.min(np.linalg.norm(np.abs(old[None,:,:] - new[:,None,:]) /
                                                   (self.DELTA_SCALE + TINY), axis=-1), axis=1)
                # New clusters should not be empty
                empty = (np.sum(pxy, axis=0) == 0)[len(old):]
                divergence[empty] = 0

                if self.DEBUG_PLOT:
                    plt.figure()
                    plt.title(T)
                    c = clusters
                    p = np.argmax(pyx, axis=0)
                    plt.scatter(X[:,0], X[:,1], c=p)
                    plt.scatter(c[:,0], c[:,1], marker='x', color='red')

                self.cluster_centers_ = old
                if np.nanmax(divergence) > 10:
                    if self.DEBUG_PLOT: plt.xlabel('keep')
                    idx = np.argmax(divergence)
                    tag = len(self.cluster_centers_)
                    self.cluster_centers_ = np.append(self.cluster_centers_, [new[idx]], axis=0)
                    self.bifurcation_tree_.create_node(tag, tag, parent=idx, data={"T": T})
            else:
                not_empty = np.sum(pxy, axis=0) != 0
                self.cluster_centers_[not_empty] = clusters[not_empty]
            # Exit loop
            if T == 0:
                break
            #  Check T > Tmin
            if T <= self.TMIN:
                T = 0
                continue
            # Cooldown
            T = self.ALPHA * T
        return self

    def predict(self, X):
        """Predict assignment probability vectors for each sample in X.

        Args:
            X (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            P (np.ndarray): Assignment probability vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers_"])
        # Your code goes here
        P = ismin(self.d_(self.cluster_centers_[None,:,:],X[:,None,:], axis=-1), axis=1).astype(float)
        return P

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.

        Args:
            X (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers_"])

        # Your code goes here
        Y = self.d_(self.cluster_centers_[None,:,:],X[:,None,:], axis=-1)
        return Y

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting"""
        check_is_fitted(self, ["bifurcation_tree_"])

        # Your code goes here

        return None

    def plot_fit(self):
        check_is_fitted(self, ["cluster_centers_"])
        try:
            dist = np.array(self.log['distortion'][:-2])
            temp = np.array(self.log['temperature'][:-2])
            n_c  = np.array(self.log['n_effective_clusters'][:-2])
        except KeyError:
            raise KeyError("Logging must be enabled for plotting")
        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots(3,1)
        ax[0].plot(dist/(np.min(dist) + TINY), 'k--')
        ax[0].set_ylabel("<D/Dmin>")
        ax[0].set_xscale("log", nonposx='clip')
        ax[0].set_yscale("log", nonposy='clip')
        ax[1].step(n_c, 'k')
        ax[1].set_xscale("log", nonposx='clip')
        ax[1].set_ylabel("n_effective_clusters")
        ax[2].plot(temp, 'k--')
        ax[2].set_xscale("log", nonposx='clip')
        #ax[2].set_yscale("log", nonposy='clip')
        ax[2].set_ylabel("T")
        ax[2].set_xlabel("steps")
        return fig, ax

if __name__ == '__main__':
    plt.ion()
    X = np.random.normal(0,1, size=(1000,2))
    DAC = DeterministicAnnealingClustering(log={'effective_distortion', 'distortion', 'temperature', 'n_effective_clusters', 'cluster_centers'})

    try:
        DAC.fit(X)
    except KeyboardInterrupt:
        print("interrupted")

    c = DAC.cluster_centers_
    p = np.argmax(DAC.predict(X), axis=-1)
    fig = plt.figure(figsize=(20,10))
    ax = fig.subplots(1,2)
    ax[0].scatter(X[:,0], X[:,1])
    ax[1].scatter(X[:,0], X[:,1], c=p)
    ax[1].scatter(c[:,0], c[:,1], marker='x', color='red')
    plt.show()
    DAC.plot_fit()
    input()
