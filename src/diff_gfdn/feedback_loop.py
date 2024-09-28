from typing import Optional

import torch
import torch.nn as nn

from .config.config import CouplingMatrixType
from .utils import matrix_convolution, to_complex


class Skew(nn.Module):

    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)


class MatrixExponential(nn.Module):

    def forward(self, X):
        return torch.matrix_exp(X)


class ND_Rotation(nn.Module):
    """Givens rotation matrix for N-D rotations, each parameterised by a rotation angle"""

    def givens_rotation_2D(self, angle: float) -> torch.tensor:
        """
        Given an angle in radians,
        return the 2x2 Givens rotation matrix
        """
        return torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                             [torch.sin(angle),
                              torch.cos(angle)]])

    def forward(self, alpha: torch.tensor) -> torch.tensor:
        N = len(alpha) + 1
        givens_matrix = torch.eye(N)

        for i in range(N - 1):
            sub_matrix = self.givens_rotation_2D(alpha[i])
            cur_matrix = torch.eye(N)
            cur_matrix[i:i + 2, i:i + 2] = sub_matrix
            givens_matrix = torch.mm(givens_matrix, cur_matrix)
        return givens_matrix


class FIRParaunitary(nn.Module):
    """Construct FIR paraunitary matrix from unit norm vectors"""

    def __init__(self, N: int, order: int):
        self.N = N
        self.order = order

    def construct_elementary_householder_matrix(
            self, unit_vector: torch.tensor) -> torch.tensor:
        """
        Construct elementary order-1 householder matrix of the form
        I - (1-z^{-1}) vv^H
        """
        assert len(unit_vector) == self.N
        householder_matrix = torch.outer(unit_vector, unit_vector)
        elementary_householder_matrix = torch.zeros((self.N, self.N, 2))
        elementary_householder_matrix[..., 0] = torch.eye(
            self.N) - householder_matrix
        elementary_householder_matrix[..., 1] = householder_matrix
        return elementary_householder_matrix

    def forward(self, unitary_matrix: torch.tensor,
                unit_vectors: torch.tensor) -> torch.tensor:
        """
        Get a PU matrix by cascading a series of order 1 householder transformations
        Args:
            unitary_matrix (torch.tensor): N x N unitary matrix for the zeroth order polynomial
            unit_vectors (torch.tensor): N x order-1 matrix containing unit vectors in each column
        Returns:
            torch.tensor: N x N x order paraunitary matrix
        """
        assert unitary_matrix.shape == (self.N, self.N)
        assert unit_vectors.shape == (self.N, self.order - 1)

        # this is going to be the polynomial matrix we construct
        poly_matrix = unitary_matrix[..., None]

        for k in range(self.order - 1):
            cur_householder_matrix = self.construct_elementary_householder_matrix(
                unit_vectors[:, k], self.N)
            poly_matrix = matrix_convolution(cur_householder_matrix,
                                             poly_matrix)

        return poly_matrix


class FeedbackLoop(nn.Module):

    def __init__(self,
                 num_groups: int,
                 num_delay_lines_per_group: int,
                 delays: torch.tensor,
                 gains: torch.tensor,
                 coupling_matrix_type: CouplingMatrixType,
                 coupling_matrix_order: Optional[int] = None):
        """
        Class implementing the feedback loop of the FDN (D_m(z) - A(z)Gamma(z))^{-1}
        Args:
            num_groups (int): number of groups in the GFDN
            num_delay_lines_per_group (int): number of delay lines in each group
            delays (List): delay line lengths in samples
            gains (List): delay line absorption gains
            coupling_matrix_type (CouplingMatrixType): scalar or filter coupling
        """
        self.num_groups = num_groups
        self.num_delay_lines_per_group = num_delay_lines_per_group
        self.delays = self.delays
        self.num_delays = len(self.delays)
        self.absorption_gains = gains
        self._eps = 1e-9

        # initialise the feedback matrices, which are learnable
        # these are individual feedback matrices for each group that
        # are orthonormal and of size N=Ndel/Ngroups, each distributed
        # uniformly in (-1/sqrt(N), +1/sqrt(N))
        self.M = nn.Parameter(
            (2 * torch.rand(self.num_groups, self.num_delay_lines_per_group,
                            self.num_delay_lines_per_group) - 1) /
            torch.sqrt(self.num_delay_lines_per_group))

        # orthonormal parameterisation of the matrices in M
        self.ortho_param = nn.Sequential(Skew(), MatrixExponential())

        if coupling_matrix_type == CouplingMatrixType.SCALAR:
            # Nroom - 1 rotation angles for getting an Nroom x Nroom unitary coupling matrix
            self.alpha = nn.Parameter(
                (2 * torch.rand(self.num_groups - 1) - 1) / (0.25 * np.pi))
            self.nd_rotation = ND_Rotation()

        elif coupling_matrix_type == CouplingMatrixType.FILTER:
            self.coupling_matrix_order = coupling_matrix_order
            # order - 1 unit norm householder vectors
            self.unit_vectors = nn.Parameter(
                torch.randn(self.num_groups, self.num_groups,
                            self.coupling_matrix_order - 1))

            self.unitary_matrix = nn.Parameter(
                (2 * torch.rand(self.num_groups, self.num_groups) - 1) /
                torch.sqrt(self.num_groups))

            self.fir_paraunitary = FIRParaunitary(self.num_groups,
                                                  self.coupling_matrix_order)

    def forward(self, z: torch.tensor) -> torch.tensor:
        """Compute (D_m(z^{-1}) - A(z)Gamma(z))^{-1}"""

        # this will be of size num_freq_points x Ndel x Ndel
        D = torch.diag_embed(torch.unsqueeze(z, dim=-1)**self.delays)

        # this is of size Ndel x Ndel
        Gamma = to_complex(torch.diag(self.absorption_gains**self.delays))

        # get coupled feedback matrix of size Ndel x Ndel x num_freq_points
        self.coupled_feedback_matrix = self.get_coupled_feedback_matrix()

        # this should be of size num_freq_pts x Ndel x Ndel
        if self.coupling_matrix_type == CouplingMatrixType.SCALAR:
            A = torch.einsum('k, mn -> kmn', z, self.coupling_feedback_matrix)
        else:
            # view converts z into a 2D matrix of size num_freq_pts x 1,
            # but it does not copy the data, hence z ultimately remains unaffected.
            # the exponetiation converts it to order x num_freq_points matrix x order,
            # and permute converts it to order x num_freq_pts

            A = torch.einsum(
                'jim, mn -> jimn', self.coupling_feedback_matrix, (z.view(
                    -1,
                    1)**-torch.arange(0, self.coupling_matrix_order)).permute(
                        1, 0))
            A = torch.sum(A, dim=2).permute(2, 0, 1)

        # A(z) Gamma(z)
        Adecay = torch.einsum('kmn, np -> kmp', A, Gamma)
        # the inverse will be taken along the last 2 dimensions
        # the size is num_freq_pts x Ndel x Ndel
        return torch.linalg.inv(D - Adecay)

    def construct_block_mixing_matrix(self):
        """Form a block matrix with the individual mixing matrices and the coupling matrix"""
        block_M = torch.zeros((self.num_delays, self.num_delays))
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                block_M[i * self.num_delay_lines_per_group:(i + 1) *
                        self.num_delay_lines_per_group,
                        j * self.num_delay_lines_per_group:(j + 1) *
                        self.num_delay_lines_per_group] = torch.mm(
                            self.ortho_param(self.M[i]),
                            self.ortho_param(self.M[j]))

        return block_M

    def construct_coupling_matrix(self):
        """Construct the Nroom x Nroom coupling matrix"""
        if self.coupling_matrix_type == CouplingMatrixType.SCALAR:
            # N-D rotation matrix that is parameterised by rotation angles
            alpha = self.alpha.clamp(min=-0.25 * np.pi, max=0.25 * np.pi)
            phi = self.nd_rotation(alpha)

        elif coupling_matrix_type == CouplingMatrixType.FILTER:
            # generate householder matrix from unitary vector
            # ensure that the vectors have unit norm
            unit_vectors = self.unit_vectors / (
                torch.norm(self.unit_vectors, dim=0, keepdim=True) + self._eps)
            phi = self.fir_paraunitary(self.ortho_param(self.unitary_matrix),
                                       unit_vectors)
        return phi

    def get_coupled_feedback_matrix(self) -> torch.tensor:
        """
        Construct the coupled feedback matrix
        Returns:
            torch.tensor of size Num_delays x Num_delays or Num_delays x Num_delays x order
        """
        # get the block matrix of individual rooms
        block_M = self.construct_block_mixing_matrix()
        # get the coupling matrix
        self.phi = self.construct_coupling_matrix()
        # get a matrix of all ones of size num_delay_lines_per_group
        ones_matrix = torch.ones(
            (self.num_delay_lines_per_group, self.num_delay_lines_per_group))

        # get the coupling matrix
        if self.coupling_matrix_type == CouplingMatrixType.SCALAR:
            # computed as A = M_block circ (Phi otimes 1)
            coupled_feedback_matrix = block_M * torch.kron(
                self.phi, ones_matrix)

        elif coupling_matrix_type == CouplingMatrixType.FILTER:
            # computed as A[k] = M_block circ (Phi[k] otimes 1)
            coupled_feedback_matrix = torch.zeros(
                (self.num_delays, self.num_delays, self.coupling_matrix_order))
            for k in range(self.coupling_matrix_order):
                coupled_feedback_matrix[..., k] += block_M * torch.kron(
                    self.phi[..., k], ones_matrix)

        return to_complex(coupled_feedback_matrix)

    def print(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self):
        M = [self.ortho_param(self.M[i]) for i in range(self.num_groups)]
        Phi = self.phi
        L0 = self.ortho_param(self.unitary_matrix)
        unit_vectors = self.unit_vectors / (
            torch.norm(self.unit_vectors, dim=0, keepdim=True) + self._eps)
        coupled_feedback_matrix = self.get_coupled_feedback_matrix()

        return (M, Phi, L0, unit_vectors, coupled_feedback_matrix)

    @torch.no_grad()
    def get_param_dict(self):
        param_np = {}
        param_np['coupling_matrix'] = self.phi.squeeze().cpu().numpy()
        param_np['individual_mixing_matrix'] = self.M.squeeze().cpu().numpy()
        param_np[
            'coupled_feedback_matrix'] = self.coupled_feedback_matrix.squeeze(
            ).cpu().numpy()
        param_np['unitary_matrix'] = self.unitary_matrix.squeeze().cpu().numpy(
        )
        param_np['unit_vectors'] = self.unit_vectors.squeeze().cpu().numpy()
        return param_np
