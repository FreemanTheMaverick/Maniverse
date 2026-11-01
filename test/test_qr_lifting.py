"""Numerically test the thin-QR lifting algorithm from Kaneko et al. (2013)."""

import unittest

import numpy as np


def qr_lift_from_cross_gram(M):
    """Find upper-triangular R with positive diagonal: M R + R.T M.T = 2 I."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    R = np.zeros_like(M)
    for i in range(1, n + 1):
        rhs = np.zeros(i)
        for j in range(i - 1):
            rhs[j] = -M[i - 1, :j + 1] @ R[:j + 1, j]
        rhs[i - 1] = 1.0
        column = np.linalg.solve(M[:i, :i], rhs)
        R[:i, i - 1] = column
    return R


def positive_diagonal_qr(matrix):
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs, signs[:, None] * r


class QRLiftingTests(unittest.TestCase):
    def test_equation_for_generic_near_identity_matrices(self):
        rng = np.random.default_rng(20261101)
        for n in range(1, 9):
            for _ in range(10):
                M = np.eye(n) + 0.05 * rng.standard_normal((n, n))
                R = qr_lift_from_cross_gram(M)

                np.testing.assert_allclose(
                    M @ R + R.T @ M.T,
                    2.0 * np.eye(n),
                    rtol=1e-11,
                    atol=1e-11,
                )
                np.testing.assert_allclose(R, np.triu(R), atol=0.0)
                self.assertTrue(np.all(np.diag(R) > 0.0))

    def test_lift_inverts_thin_qr_retraction(self):
        rng = np.random.default_rng(12345)
        for rows, cols in ((7, 4), (5, 2), (4, 4)):
            X, _ = positive_diagonal_qr(rng.standard_normal((rows, cols)))
            W = rng.standard_normal((rows, cols))
            XtW = X.T @ W
            V = W - X @ (0.5 * (XtW + XtW.T))
            V *= 0.03

            Q, expected_R = positive_diagonal_qr(X + V)
            M = X.T @ Q
            R = qr_lift_from_cross_gram(M)
            lifted_V = Q @ R - X

            np.testing.assert_allclose(R, expected_R, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                X.T @ lifted_V + lifted_V.T @ X,
                0.0,
                atol=1e-10,
            )
            recovered_Q, _ = positive_diagonal_qr(X + lifted_V)
            np.testing.assert_allclose(recovered_Q, Q, rtol=1e-10, atol=1e-10)

if __name__ == "__main__":
    unittest.main()
