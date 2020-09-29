# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from scipy.linalg import orthogonal_procrustes

from .base import BaseAlign


class OrthogonalProcrustes(BaseAlign):
    """
    Computes the matrix solution of the classical orthogonal Procrustes [1]_
    problem, which is that given two matrices `X` and `Y` of equal shape
    (n, d), find an orthogonal matrix `self.Q_` that most closely maps `X` to
    `Y`. Subsequently, uses that matrix to transform either the original `X`,
    or a different dataset that has the same generating distribution.

    Note that when used to match two datasets, this method unlike
    :class:`~graspy.align.SeedlessProcrustes`, not only requires that the
    datasets have the same number of entries, but also that there is some
    correspondence between the entries. In graph embeddings, this usually
    corresponds to the assumption that the vertex i in graph X has the same
    latent position as the vertex i in graph Y.

    Attributes
    ----------
        Q_ : array, size (d, d)
              Final orthogonal matrix, used to modify `X`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    .. [2] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    Notes
    -----
    Formally, minimizes :math:`|| X Q - Y ||_F`, which has a closed form
    solution, whenever :math:`Q` is constrained to be an orthogonal matrix,
    that is a matrix that satisfies :math:`Q^T Q = Q Q^T = I`. For the more
    details, including the proof of the closed-form solution see [1]_.

    Implementation-wise, this class is a wrapper of the
    `scipy.linalg.orthogonal_procrustes`, which itself uses an algorithm
    described in find the optimal solution algorithm [2]_.

    """

    def __init__(
        self,
    ):

        super().__init__()

    def fit(self, X, Y):
        """
        Uses the two datasets to learn the matrix `self.Q_` that aligns the
        first dataset with the second.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions and entries as ones in `Y`.

        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            dimensions and entries as ones in `X`.

        Returns
        -------
        self : returns an instance of self

        """
        X, Y = self._check_datasets(X, Y)

        _, d = X.shape
        if X.shape[0] != Y.shape[0]:
            msg = (
                "Two datasets have different number of entries! "
                "OrthogonalProcrustes assumes that entries of the two "
                "datasets are matched. consider using SeedlessProcrustes "
                "instead."
            )
            raise ValueError(msg)

        _, d = X.shape
        self.Q_, _ = orthogonal_procrustes(X, Y)
        return self

    def fit_transform(self, X, Y):
        """
        Uses the two datasets to learn the matrix `self.Q_` that aligns the
        first dataset with the second. Then, transforms the first dataset `X`
        using the learned matrix `self.Q_`.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions and entries as ones in `Y`.

        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            dimensions and entries as ones in `X`.

        Returns
        -------
        X_prime : np.ndarray, shape (n, d)
            First dataset of vectors, aligned to second. Equal to
            `X` @ `self.Q_`.
        """
        return super().fit_transform(X, Y)
