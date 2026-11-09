"""Implicit local Gaussian state covariance, using neighbor-form curvature products."""
from __future__ import annotations

from typing import Literal

import numpy as np


class ContinuousStateCovariance:
    """Inverse Gauss-Newton curvature over [sensory potentials z, hidden states h].

    Arrays and results stay on the model backend. This is an instantaneous local
    approximation, not a temporally propagated posterior or a covariance of weights.
    """

    MAX_MARGINAL_NODES = 64

    def __init__(
        self, *, hidden_jacobian, sensory_jacobian, observation_curvature,
        hidden_prior_precision: float, sensory_prior_precision: float,
        hidden_residual_precision, coupling_precision, backend,
        stage: Literal["forecast", "posterior"],
    ):
        self.backend = backend
        self.stage = stage
        self._hidden_jacobian = hidden_jacobian
        self._sensory_jacobian = sensory_jacobian
        self._observation_curvature = observation_curvature
        self._hidden_prior_precision = hidden_prior_precision
        self._sensory_prior_precision = sensory_prior_precision
        self.sensory_nodes, self.hidden_nodes = sensory_jacobian.shape
        self.total_nodes = self.sensory_nodes + self.hidden_nodes
        xp = backend
        self._hidden_residual_precision = xp.broadcast_to(
            xp.asarray(hidden_residual_precision), (self.hidden_nodes,),
        ).copy()
        self._coupling_precision = xp.broadcast_to(
            xp.asarray(coupling_precision), (self.sensory_nodes,),
        ).copy()
        self._diagonal = xp.concatenate((
            sensory_prior_precision + self._coupling_precision + observation_curvature,
            hidden_prior_precision
            + xp.sum(self._hidden_residual_precision[:, None] * hidden_jacobian**2, axis=0)
            + xp.sum(self._coupling_precision[:, None] * sensory_jacobian**2, axis=0),
        ))
        if not bool(xp.all(xp.isfinite(self._diagonal) & (self._diagonal > 0))):
            raise ValueError("state curvature requires finite positive precision")
        self._sensory_schur_cholesky = None
        self._whitened_sensory_jacobian = None

    def sensory_information_gain(self, indices, added_precision) -> float:
        """Expected joint-state information from independent linearized sensory observations.

        Gaussian mutual information is the expected posterior/prior KL:
        https://statproofbook.github.io/P/mvn-mi.html
        For P = inv(H), observing with curvature Q gives
        I = (logdet(H + Q) - logdet(H))/2, in nats. This is exact for the
        captured local Gaussian, independent of the hypothetical measured value.

        Write H = [[D, -diag(q) B], [-B.T diag(q), E]], where D is diagonal.
        Its hidden Schur complement is S = E - B.T diag(q**2/D) B.
        Adding sensory diagonal d changes S by B.T diag(q**2*d/(D*(D+d))) B.
        Cholesky-whitening that positive update evaluates the determinant ratio
        using only hidden-by-hidden matrices, retaining correlations across the
        entire observed patch and through all unobserved states.
        """
        xp = self.backend
        indices = xp.asarray(indices)
        added = xp.asarray(added_precision, dtype=xp.float64)
        if (
            indices.ndim != 1 or indices.dtype.kind not in "iu"
            or added.shape != indices.shape
            or not bool(xp.all((indices >= 0) & (indices < self.sensory_nodes)))
            or int(xp.unique(indices).size) != indices.size
            or not bool(xp.all(xp.isfinite(added) & (added >= 0)))
        ):
            raise ValueError("observations require distinct sensory indices and finite non-negative precisions")
        if not indices.size:
            return 0.0
        diagonal = self._diagonal[:self.sensory_nodes]
        if self._sensory_schur_cholesky is None:
            q = self._coupling_precision
            unconstrained = self._sensory_prior_precision + self._observation_curvature
            schur = (
                self._hidden_prior_precision * xp.eye(self.hidden_nodes)
                + self._hidden_jacobian.T
                @ (self._hidden_residual_precision[:, None] * self._hidden_jacobian)
                + self._sensory_jacobian.T
                @ ((q * (unconstrained / diagonal))[:, None] * self._sensory_jacobian)
            )
            self._sensory_schur_cholesky = xp.linalg.cholesky(schur)
            self._whitened_sensory_jacobian = xp.linalg.solve(
                self._sensory_schur_cholesky, self._sensory_jacobian.T,
            ).T
        local_diagonal = diagonal[indices]
        ratio = added / local_diagonal
        # This bounded fraction avoids forming q**2 or subtracting nearby log determinants.
        fraction = ratio / (1.0 + ratio)
        scale = self._coupling_precision[indices] * xp.sqrt(fraction / local_diagonal)
        whitened = self._whitened_sensory_jacobian[indices] * scale[:, None]
        factor = xp.linalg.cholesky(
            xp.eye(self.hidden_nodes) + whitened.T @ whitened,
        )
        gain = float(0.5 * xp.sum(xp.log1p(ratio)) + xp.sum(xp.log(xp.diag(factor))))
        if not np.isfinite(gain) or gain < 0:
            raise FloatingPointError("sensory information gain must be finite and non-negative")
        return gain

    @property
    def precision_diagonal(self):
        """Return diag(H); its reciprocal is not the marginal state variance."""
        return self._diagonal.copy()

    def _vector(self, values):
        vector = self.backend.asarray(values, dtype=self.backend.float64)
        if vector.shape != (self.total_nodes,) or not bool(self.backend.all(self.backend.isfinite(vector))):
            raise ValueError(f"state vector must have {self.total_nodes} finite entries")
        return vector

    def _precision_product(self, vector):
        xp = self.backend
        z, h = vector[:self.sensory_nodes], vector[self.sensory_nodes:]
        hidden_change = self._hidden_jacobian @ h
        coupling_change = z - self._sensory_jacobian @ h
        # Each residual sends its precision-weighted change back through its own Jacobian.
        return xp.concatenate((
            (self._sensory_prior_precision + self._observation_curvature) * z
            + self._coupling_precision * coupling_change,
            self._hidden_prior_precision * h
            + self._hidden_jacobian.T @ (self._hidden_residual_precision * hidden_change)
            - self._sensory_jacobian.T @ (self._coupling_precision * coupling_change),
        ))

    def precision_product(self, vector):
        """Apply the sum of local J.T @ precision @ J contributions without forming H."""
        return self._precision_product(self._vector(vector))

    def covariance_product(self, vector, *, rtol: float = 1e-10, max_iterations: int | None = None):
        """Solve H x = vector with diagonally preconditioned conjugate gradients."""
        if (
            isinstance(rtol, bool) or not isinstance(rtol, (int, float, np.number))
            or not np.isfinite(rtol) or not 0 < rtol < 1
        ):
            raise ValueError("rtol must be finite and between zero and one")
        if max_iterations is None:
            max_iterations = 4 * self.hidden_nodes + 16
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        xp = self.backend
        rhs = self._vector(vector)
        solution = xp.zeros_like(rhs)
        rhs_norm = float(xp.linalg.norm(rhs))
        if rhs_norm == 0:
            return solution
        tolerance = float(rtol) * rhs_norm
        residual = rhs.copy()
        # This diagonal speeds the solve; it is not an approximation to the covariance.
        preconditioned = residual / self._diagonal
        direction = preconditioned.copy()
        residual_product = float(residual @ preconditioned)
        for _ in range(max_iterations):
            curvature_direction = self._precision_product(direction)
            denominator = float(direction @ curvature_direction)
            if not np.isfinite(denominator) or denominator <= 0:
                raise RuntimeError("state covariance solve encountered non-positive curvature")
            step = residual_product / denominator
            solution += step * direction
            residual -= step * curvature_direction
            if float(xp.linalg.norm(residual)) <= tolerance:
                # Check the original equation so recurrence roundoff cannot report false convergence.
                residual = rhs - self._precision_product(solution)
                if float(xp.linalg.norm(residual)) <= tolerance:
                    return solution
                preconditioned = residual / self._diagonal
                direction = preconditioned.copy()
                residual_product = float(residual @ preconditioned)
                continue
            preconditioned = residual / self._diagonal
            next_product = float(residual @ preconditioned)
            direction = preconditioned + (next_product / residual_product) * direction
            residual_product = next_product
        raise RuntimeError(f"state covariance solve did not converge in {max_iterations} iterations")

    def marginal_covariance(self, indices):
        """Return a bounded block of the joint covariance, including paths through omitted states."""
        indices = np.asarray(indices)
        if (
            indices.ndim != 1 or not 1 <= len(indices) <= self.MAX_MARGINAL_NODES
            or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= self.total_nodes) or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                f"indices must select 1..{self.MAX_MARGINAL_NODES} distinct neural states"
            )
        xp = self.backend
        selected = xp.asarray(indices)
        block = xp.empty((len(indices), len(indices)), dtype=xp.float64)
        basis = xp.zeros(self.total_nodes, dtype=xp.float64)
        # Solve one column at a time: neither a full inverse nor an N-by-query workspace is retained.
        for column, index in enumerate(indices):
            basis.fill(0)
            basis[int(index)] = 1
            block[:, column] = self.covariance_product(basis)[selected]
        return block
