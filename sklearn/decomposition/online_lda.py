"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: http://www.cs.princeton.edu/~mdhoffma/code/onlineldavb.tar
"""

# Author: Chyi-Kwei Yau
# Author: Matthew D. Hoffman (original onlineldavb implementation)

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from ..base import BaseEstimator, TransformerMixin
from ..utils import (check_random_state, check_array,
                     gen_batches, gen_even_slices)
from ..utils.validation import NotFittedError

from ..externals.joblib import Parallel, delayed, cpu_count
from ..externals.six.moves import xrange

from ._online_lda import (mean_change, _dirichlet_expectation_1d,
                          _dirichlet_expectation_2d)


def _dirichlet_expectation(X):
    """
    For an array theta ~ Dir(X), computes `E[log(theta)]` given X.

    Parameters
    ----------
    X : array-like
        1 or 2 dimention vector

    Returns
    -------
    dirichlet_expect: array-like
        Dirichlet expectation of input array X
    """

    if len(X.shape) == 1:
        dirichlet_expect = _dirichlet_expectation_1d(X)
    else:
        dirichlet_expect = _dirichlet_expectation_2d(X)
    return dirichlet_expect


def _update_doc_distribution(X, exp_topic_word_distr, doc_topic_prior, rng, max_iters,
                             mean_change_tol, cal_diff):
    """
    E-step: update document topic distribution.
    (In literature, it is latent variable `gamma`)

    Parameters
    ----------
    X : sparse matrix, shape = [n_samples, n_features]
        Document word matrix.

    exp_topic_word_distr : dense matrrix, shape = [n_topics, n_features]
        Exponential value of expection of log topic word distribution.
        In literature, it is `exp(E[log(beta)])`.

    doc_topic_prior : float
        Prior of document topic distribution `theta`.

    rng : int or RandomState instance or None, optional (default: None)
        Pseudo Random Number generator seed control.

    max_iters : int
        Max number of iterations for updating document topic distribution in E-step.

    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-setp.

    cal_diff : boolean
        Parameter that indicate to calculate differene in `component_` or not.
        Set `cal_diff` to `True` when we need to run M-step.

    Returns
    -------
    (doc_topic_distr, component_diff) :
        `doc_topic_distr` is unnormailzed topic distribution for each document.
        In literature, it is `gamma` latent variable. we can calcuate `E[log(theta)]`
        from it.
        `component_diff` is the difference of `component_` when `cal_diff` is True.
        Otherwise, it is None.

    """

    n_samples, n_features = X.shape
    n_topics = exp_topic_word_distr.shape[0]

    # this is variable `gamma` in literature
    doc_topic_distr = rng.gamma(100., 1. / 100., (n_samples, n_topics))
    # this is `exp(E[log(theta)])` in literature
    exp_doc_topic = np.exp(_dirichlet_expectation(doc_topic_distr))

    # diff on `component_` (only calculate it when `cal_diff` is True)
    component_diff = np.zeros(exp_topic_word_distr.shape) if cal_diff else None

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    for d in xrange(n_samples):
        ids = X_indices[X_indptr[d]:X_indptr[d + 1]]
        cnts = X_data[X_indptr[d]:X_indptr[d + 1]]

        doc_topic_d = doc_topic_distr[d, :]
        exp_doc_topic_d = exp_doc_topic[d, :]
        exp_topic_word_d = exp_topic_word_distr[:, ids]

        # The optimal phi_{dwk} is proportional to 
        # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
        norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + 1e-100

        # Iterate between `doc_topic_d` and `norm_phi` until convergence
        for it in xrange(0, max_iters):
            last_d = doc_topic_d

            doc_topic_d = doc_topic_prior + exp_doc_topic_d * \
                np.dot(cnts / norm_phi, exp_topic_word_d.T)
            exp_doc_topic_d = np.exp(_dirichlet_expectation(doc_topic_d))
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + 1e-100

            meanchange = mean_change(last_d, doc_topic_d)
            if meanchange < mean_change_tol:
                break
        doc_topic_distr[d, :] = doc_topic_d

        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_diff:
            component_diff[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)

    return (doc_topic_distr, component_diff)


class LatentDirichletAllocation(BaseEstimator, TransformerMixin):

    """
    Online Latent Dirichlet Allocation implementation with variational inference

    References
    ----------
    [1] "Online Learning for Latent Dirichlet Allocation", Matthew D. Hoffman, 
        David M. Blei, Francis Bach

    [2] Matthew D. Hoffman's onlineldavb code. Link:
        http://www.cs.princeton.edu/~mdhoffma/code/onlineldavb.tar


    Parameters
    ----------
    n_topics : int, optional (default: 10)
        Number of topics.

    doc_topic_prior : float, optional (defalut: 0.1)
        Prior of document topic distribution `theta`. In general, it is `1 / n_topics`.
        In literature, it is Hyperparameter parameter `alpha`.

    topic_word_prior : float, optional (default: 0.1)
        Prior of topic word distribution `beta`. In general, it is `1 / n_topics`.
        In literature, it is hyperparameter parameter `eta`.

    learning_decay : float, optional (default: 0.7)
        It is a parameter that control learning weight for `_component` in online learning.
        The value should be set between (0.5, 1.0] to guarantee asymptotic convergence.
        When the value is 0.0 and batch_size is `n_samples`, the udpate is same as batch learning.
        In literature, it is parameter `kappa`.

    learning_offset : float, optional (default: 1000.)
        A (positive) parameter that downweights early iterations in online learning.
        It should be greater than 1.0. In literature, it is parameter `tau0`.        

    n_samples : int, optional (default: 1e6)
        Total umber of document. It is only used in online learing.
        In batch learning, n_samples is set to X.shape[0]

    batch_size : int, optional (default: 128)
        Number of document to udpate in each EM-step

    evaluate_every : int optional (default: 5)
        How many iterations to evaluate perplexity once. Only used in batch learning yet.
        set it to -1 to not evalute perplexity in training at all.
        Evaluating perplexity in every iteration will increase training time up to 2X.

    normalize_doc : boolean, optional (default: False)
        Normalize the topic distribution for each document or not.
        If `True`, sum of topic distribution will be 1.0 for each document.

    perp_tol : float, optional (default: 1e-1)
        Perplexity tolerance in batch learning.

    mean_change_tol : float, optional (default: 1e-3)
        Stopping tolerance for updating document topic distribution in E-setp.

    max_doc_update_iter : int (default: 100)
        Max number of iterations for updating document topic distribution in E-step.

    n_jobs : int, optional (default: 1)
        Number of parallel jobs to run in E-step. -1 for autodetect.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState instance or None, optional (default: None)
        Pseudo Random Number generator seed control.


    Attributes
    ----------
    components_ : array, [n_topics, n_features]
        Topic word distribution. components_[i, j] represents word `j` in topic `i`.
        In literature, it is latent parameter `lambda`, and we can calcuate 
        `E[log(beta)]` from it.

    n_iter_ : int
        Number of iteration.

    """

    def __init__(self, n_topics=10, doc_topic_prior=.1, topic_word_prior=.1,
                 learning_decay=.7, learning_offset=1000., batch_size=128,
                 evaluate_every=5, n_samples=1e6, normalize_doc=False,
                 perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=1, verbose=0, random_state=None):
        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.n_samples = n_samples
        self.normalize_doc = normalize_doc
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _init_latent_vars(self, n_features):
        """
        Initialize latent variables.
        """
        self.rng = check_random_state(self.random_state)
        self.n_iter_ = 1
        self.n_features = n_features
        init_gamma = 100.
        init_var = 1. / init_gamma

        # In literature, this is variable `lambda`
        self.components_ = self.rng.gamma(
            init_gamma, init_var, (self.n_topics, n_features))
        # In literature, this is `E[log(beta)]`
        self.dirichlet_component_ = _dirichlet_expectation(self.components_)
        # In literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(self.dirichlet_component_)

    def _e_step(self, X, cal_diff):
        """
        E-step

        parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            Document word matrix.

        cal_diff : boolean
            parameter that indicate to calculate differene in `component_` or not.
            Set `cal_diff` to `True` when we need to run M-step.

    Returns
    -------
    (doc_topic_distr, component_diff) :
        `doc_topic_distr` is unnormailzed topic distribution for each document.
        In literature, it is `gamma` variable. we can calcuate `E[log(theta)]`
        from it.
        `component_diff` is the difference of `component_` when `cal_diff == True`.
        Otherwise, it is None.

        """

        # parell run e-step
        if self.n_jobs == -1:
            n_jobs = cpu_count()
        else:
            n_jobs = self.n_jobs

        results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_update_doc_distribution)
            (X[idx_slice, :], self.exp_dirichlet_component_, self.doc_topic_prior,
             self.rng, self.max_doc_update_iter, self.mean_change_tol, cal_diff)
            for idx_slice in gen_even_slices(X.shape[0], n_jobs))

        # merge result
        doc_topics, comp_diffs = zip(*results)
        doc_topic_distr = np.vstack(doc_topics)

        if cal_diff:
            # This step finishes computing the sufficient statistics for the M step
            component_diff = np.zeros(self.components_.shape)
            for comp_diff in comp_diffs:
                component_diff += comp_diff
            component_diff *= self.exp_dirichlet_component_
        else:
            component_diff = None

        return (doc_topic_distr, component_diff)

    def _em_step(self, X, batch_update):
        """
        EM update for 1 iteration
        update `_component` by bath VB or online VB

        parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            Document word matrix.

        batch_update : boolean
            parameter that control updating method.
            `True` for batch learning, `False` for online learning

        Returns
        -------
        doc_topic_distr : array, shape = [n_samples, n_topics]
            Unnormailzed document topic distribution.
        """

        # E-step
        doc_topic_distr, component_diff = self._e_step(X, cal_diff=True)

        # M-step
        if batch_update:
            self.components_ = self.topic_word_prior + component_diff
        else:
            # online update
            # In literature, the weight is `rho`
            weight = np.power(self.learning_offset + self.n_iter_, -self.learning_decay)
            doc_ratio = float(self.n_samples) / X.shape[0]
            self.components_ *= (1 - weight)
            self.components_ += (weight *
                                 (self.topic_word_prior + doc_ratio * component_diff))

        # update `component_` related variables 
        self.dirichlet_component_ = _dirichlet_expectation(self.components_)
        self.exp_dirichlet_component_ = np.exp(self.dirichlet_component_)
        self.n_iter_ += 1
        return doc_topic_distr

    def _to_csr(self, X):
        """
        check & convert X to csr format

        parameters
        ----------
        X :  array-like
        
        """
        X = check_array(X, accept_sparse='csr')
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        return X

    def fit_transform(self, X, y=None, max_iters=10):
        """
        Learn a model for X and returns the transformed data

        Parameters
        ----------
        X : array or sparse matrix, shape = [n_samples, n_features]
            Document word matrix.

        max_iters : int, (default: 10)
            Max number of iterations

        Returns
        -------
        doc_topic_distr : array, [n_samples, n_topics]
            Topic distribution for each document.
        """

        X = self._to_csr(X)
        return self.fit(X, max_iters).transform(X)

    def partial_fit(self, X, y=None):
        """
        Online Learning with Min-Batch update

        Parameters
        ----------
        X : array or sparse matrix, shape = [n_samples, n_features]
            Document word matrix.

        Returns
        -------
        self
        """

        X = self._to_csr(X)
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        # initialize parameters or check
        if not hasattr(self, 'components_'):
            self._init_latent_vars(n_features)

        if n_features != self.n_features:
            raise ValueError(
                "feature dimension(vocabulary size) doesn't match.")

        for idx_slice in gen_batches(n_samples, batch_size):
            self._em_step(X[idx_slice, :], batch_update=False)

        return self

    def fit(self, X, y=None, max_iters=10):
        """
        Learn model from X. This function is for batch learning,
        so it will override the old _component variables.

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            Document word matrix.
        
        max_iters : int, (default: 10)
            Max number of iterations

        Returns
        -------
        self
        """

        X = self._to_csr(X)
        n_samples, n_features = X.shape
        evaluate_every = self.evaluate_every

        # initialize parameters
        self._init_latent_vars(n_features)

        # change to perplexity later
        last_bound = None
        for i in xrange(max_iters):
            doc_topics = self._em_step(X, batch_update=True)
            # check perplexity
            if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                bound = self.perplexity(X, doc_topics, sub_sampling=False)
                if self.verbose:
                    print('iteration: %d, perplexity: %.4f' % (i + 1, bound))

                if last_bound and abs(last_bound - bound) < self.perp_tol:
                    break
                last_bound = bound

        return self

    def transform(self, X):
        """
        Transform data X according to the fitted model.

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            Document word matrix.
            `n_features` must be the same as `self.n_features`

        max_iters : int, (default: 20)
            Max number of iterations.

        Returns
        -------
        doc_topic_distr : array, [n_samples, n_topics]
            Document topic distribution for X.
        """

        X = self._to_csr(X)
        n_samples, n_features = X.shape

        if not hasattr(self, 'components_'):
            raise NotFittedError(
                "no 'components_' attribute in model. Please fit model first.")
        # make sure word size is the same in fitted model and new doc
        # matrix
        if n_features != self.n_features:
            raise ValueError(
                "feature dimension(vocabulary size) does not match.")

        doc_topic_distr, _ = self._e_step(X, False)

        if self.normalize_doc:
            doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr

    def _approx_bound(self, X, doc_topic_distr, sub_sampling):
        """
        Calculate approximate bound for data X and document topic distribution
        Since log-likelihood cannot be computed directly, we use this bound
        estimate it.

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            Document word matrix.

        doc_topic_distr : array, shape = [n_samples, n_topics]
            Document topic distribution. In literature, it is `gamma`.

        sub_sampling : boolean, optional, (default: False)
            Compensate for subsampling of documents.
            It is used in calcuate bound in online learning.

        Returns
        -------
        score : float

        """
        X = self._to_csr(X)
        n_samples, n_topics = doc_topic_distr.shape
        score = 0
        dirichlet_doc_topic = _dirichlet_expectation(doc_topic_distr)

        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

        # E[log p(docs | theta, beta)]
        for d in xrange(0, n_samples):
            ids = X_indices[X_indptr[d]:X_indptr[d + 1]]
            cnts = X_data[X_indptr[d]:X_indptr[d + 1]]
            id_length = len(ids)
            norm_phi = np.zeros(id_length)
            for i in xrange(0, id_length):
                temp = dirichlet_doc_topic[d, :] + self.dirichlet_component_[:, ids[i]]
                tmax = temp.max()
                norm_phi[i] = np.log(np.sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cnts * norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self.doc_topic_prior - doc_topic_distr) * dirichlet_doc_topic)
        score += np.sum(gammaln(doc_topic_distr) - gammaln(self.doc_topic_prior))
        score += np.sum(
            gammaln(self.doc_topic_prior * self.n_topics) - gammaln(np.sum(doc_topic_distr, 1)))

        # Compensate for the subsampling of the population of documents
        # E[log p(beta | eta) - log q (beta | lambda)]
        score += np.sum((self.topic_word_prior - self.components_) * self.dirichlet_component_)
        score += np.sum(gammaln(self.components_) - gammaln(self.topic_word_prior))
        score += np.sum(gammaln(self.topic_word_prior * self.n_features)
                        - gammaln(np.sum(self.components_, 1)))

        # Compensate for the subsampling of the population of documents
        if sub_sampling:
            doc_ratio = float(self.n_samples) / n_samples
            score *= doc_ratio

        return score

    def score(self, X, y=None):
        """
        use approximate log-likelihood as score

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
             Document word matrix.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """

        X = self._to_csr(X)
        doc_topic_distr = self.transform(X)
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def perplexity(self, X, doc_topic_distr, sub_sampling=False):
        """
        calculate approximate perplexity for data X and topic distribution `gamma`.
        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            Document word matrix.

        doc_topic_distr : array, shape = [n_samples, n_topics]
            Document topic distribution.

        Returns
        -------
        score : float
            Perplexity score.
        """
        X = self._to_csr(X)
        current_samples = X.shape[0]
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)
        perword_bound = bound / np.sum(X.data)

        if sub_sampling:
            perword_bound = perword_bound * (float(current_samples) / self.n_samples)

        return np.exp(-1.0 * perword_bound)
