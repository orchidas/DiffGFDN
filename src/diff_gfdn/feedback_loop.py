from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import torch
from torch import nn

from .absorption_filters import decay_times_to_gain_per_sample
from .config.config import CouplingMatrixType
from .gain_filters import BiquadCascade, IIRFilter, SOSFilter
from .utils import matrix_convolution, to_complex

# pylint: disable=E0606, W0718


class Skew(nn.Module):
    """Return a skew symmetric matrix from matrix X"""

    def forward(self, X: torch.tensor):
        """
        Args:
            X : square real matrix
        """
        A = X.triu(1)
        return A - A.transpose(-1, -2)


class MatrixExponential(nn.Module):
    """Return the exponentiated matrix X"""

    def forward(self, X: torch.tensor):
        """
        Args:
            X : square real matrix
        """
        return torch.matrix_exp(X)


class ND_Unitary(nn.Module):
    """An N-D unitary matrix constructed with Givens' rotations"""

    def construct_planar_rotation_matrix(self, alpha: float, N: int,
                                         i: int) -> torch.tensor:
        """
        Planar rotation matrix derivative in rows i and N-1
        Args:
            alpha (float) : angle of rotation
            N (int) : order of matrix
            i (int) : position of planar rotation
        Returns:
            NDArray: N x N matrix R_i
        """
        R = torch.eye(N)
        R[i, i] = torch.cos(alpha)
        R[i, -1] = -torch.sin(alpha)
        R[-1, i] = torch.sin(alpha)
        R[-1, -1] = torch.cos(alpha)
        return R

    def forward(self, alpha: torch.tensor, N: int) -> torch.tensor:
        """
        Construct any NxN unitary matrix using,
        U_n = R_{n-2}...R_0 [[U_{n-1}, 0], [0, pm 1]]
        Args:
            alpha (tensor): list of N*(N-1) / 2 rotation angles
            N (int): size of matrix
        Returns:
            tensor : unitary matrix of size NxN
        """
        assert len(alpha) == N * (N - 1) // 2
        rot_matrix = torch.eye(N)
        if N == 1:
            return 1
        else:
            start_idx = (N - 1) * (N - 2) // 2
            cur_alpha = alpha[start_idx:]
            for i in range(N - 1):
                rot_matrix = torch.mm(
                    self.construct_planar_rotation_matrix(cur_alpha[i], N, i),
                    rot_matrix)
            # this is the matrix [[U_n-1, 0], [0, 1]]
            big_matrix = torch.eye(N)
            big_matrix[:N - 1, :N - 1] = self.forward(alpha[:start_idx], N - 1)
            result = torch.mm(rot_matrix, big_matrix)
            # all of the intermediate matrices must be unitary
            # assert is_unitary(result)[0]
            return result


class FIRParaunitary(nn.Module):
    """Construct FIR paraunitary matrix from unit norm vectors"""

    def __init__(self, N: int, order: int):
        """
        Args:
        N (int): dimensions of the matrix
        order (int): order of the polynomial matrix
        """
        super().__init__()
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
        assert unitary_matrix.shape == (self.N, self.N)  # noqa: E251
        assert unit_vectors.shape == (self.N, self.order - 1)

        # this is going to be the polynomial matrix we construct
        # add a dimension after the last axis
        poly_matrix = torch.eye(self.N)[..., None]

        for k in range(self.order - 1):
            cur_householder_matrix = self.construct_elementary_householder_matrix(
                unit_vectors[:, k])
            poly_matrix = matrix_convolution(cur_householder_matrix,
                                             poly_matrix)

        poly_matrix = matrix_convolution(
            poly_matrix, unitary_matrix.view(self.N, self.N, 1))

        return poly_matrix


class FeedbackLoop(nn.Module):

    def __init__(self,
                 sample_rate: float,
                 num_groups: int,
                 num_delay_lines_per_group: int,
                 delays: torch.tensor,
                 use_absorption_filters: bool,
                 coupling_matrix_type: CouplingMatrixType = None,
                 coupling_matrix_order: Optional[int] = None,
                 colorless_feedback_matrix: Optional[torch.tensor] = None,
                 gains: Optional[torch.Tensor] = None,
                 common_decay_times: Optional[List] = None,
                 device: torch.device = 'cpu'):
        """
        Class implementing the feedback loop of the FDN (D_m(z) - A(z)Gamma(z))^{-1}
        Args:
            num_groups (int): number of groups in the GFDN
            num_delay_lines_per_group (int): number of delay lines in each group
            delays (List): delay line lengths in samples
            gains (List): delay line absorption gains / filters, can be learnable
            common_decay_times (list, optional): list of initial common decay times (one for each room). 
                                                 If none, they are learned by network
            use_absorption_filters (bool): whether the delay line gains are gains or filters
            coupling_matrix_type (CouplingMatrixType): scalar or filter coupling
            coupling_matrix_order (optional, int): order of the PU filter coupling matrix
            colorless_feedback_matrix (torch.tensor, optional): the block diagonal
                                colorless feedback matrix obtained from ColorlessFDN optimisation
            device (torch.device): the training device, CPU or CUDA
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.num_groups = num_groups
        self.num_delay_lines_per_group = num_delay_lines_per_group
        self.delays = delays
        self.num_delays = len(self.delays)
        self.use_absorption_filters = use_absorption_filters
        self.device = device
        self._eps = 1e-9
        self.coupling_matrix_type = coupling_matrix_type
        self.coupling_matrix_order = coupling_matrix_order
        self._init_absorption(gains, common_decay_times)
        self._init_feedback_matrix(colorless_feedback_matrix)

    def _init_absorption(self,
                         gains: Optional[torch.Tensor] = None,
                         common_decay_times: Optional[List] = None):
        """
        Initialise the absorption gains/filters
        Args:
            gains : Delay line gains/filters. If None, they are learnable
        """
        if gains is None:
            if self.use_absorption_filters:
                logger.error("Cannot learn absorption filters yet")
            else:
                logger.info("Learning common decay times...")
                if common_decay_times is None:
                    lower_decay_time = 100 * 1e-3
                    upper_decay_time = 2000 * 1e-3
                    random_init_decay_times = lower_decay_time + (
                        upper_decay_time - lower_decay_time) * torch.rand(
                            self.num_groups)
                    self.common_decay_times = nn.Parameter(
                        random_init_decay_times)
                else:
                    logger.info(
                        "Initialising with known common decay times...")
                    # initialise with pre-calculated decay times and then fine-tune it
                    self.common_decay_times = nn.Parameter(
                        torch.tensor(common_decay_times.squeeze()))

                self.delays_by_group = [
                    torch.tensor(
                        self.delays[i:i + self.num_delay_lines_per_group],
                        device=self.device) for i in range(
                            0, self.num_delays, self.num_delay_lines_per_group)
                ]
                self.delay_line_gains = torch.cat([
                    decay_times_to_gain_per_sample(
                        self.common_decay_times[i], self.delays_by_group[i],
                        torch.tensor(self.sample_rate))
                    for i in range(self.num_groups)
                ]).to(self.device)
        else:
            logger.info("Using provided common decay times...")
            # whether to use absorption filters or scalar gains in delay lines
            if self.use_absorption_filters:
                logger.info("Using absorption filters")
                filter_order = gains.shape[1]
                # absorption gains as IIR filters with Prony's method
                if gains.ndim == 3:
                    self.delay_line_gains = IIRFilter(filter_order,
                                                      self.num_delays,
                                                      gains[..., 0],
                                                      gains[..., 1],
                                                      device=self.device)
                # absorption gains as SOS with GEQ fitting
                else:
                    self.delay_line_gains = []
                    for k in range(self.num_delays):
                        delay_line_biquads = BiquadCascade(
                            filter_order, gains[k, :, :, 0], gains[k, :, :, 1])
                        self.delay_line_gains.append(
                            SOSFilter(filter_order,
                                      delay_line_biquads,
                                      device=self.device))
            else:
                logger.info("Using absorption gains")
                self.delay_line_gains = gains

    def _init_feedback_matrix(self,
                              colorless_feedback_matrix: Optional[
                                  torch.tensor] = None):
        """
        Initialise feedback matrix
        Args:
            colorless_feedback_matrix: If pre-optimised feedback matrix is provided, then use it
        """
        # orthonormal parameterisation of the matrices in M
        self.ortho_param = nn.Sequential(Skew(), MatrixExponential())

        # in this case, the coupled feedback matrix is any unitary matrix. It has no
        # inherent structure
        if self.coupling_matrix_type == CouplingMatrixType.RANDOM:
            self.random_feedback_matrix = nn.Parameter(
                (2 * torch.rand(self.num_delays, self.num_delays) - 1) /
                np.sqrt(self.num_delay_lines_per_group))

        # otherwise, use coupled matrix structure proposed in GFDN papers
        # with individual mixing matrices and a coupling matrix connecting them
        else:
            # these are individual feedback matrices for each group that
            # are orthonormal and of size N=Ndel/Ngroups, each distributed
            # uniformly in (-1/sqrt(N), +1/sqrt(N))
            if colorless_feedback_matrix is not None:
                self.M = colorless_feedback_matrix.clone().detach()
            else:
                self.M = nn.Parameter((2 * torch.rand(
                    self.num_groups, self.num_delay_lines_per_group,
                    self.num_delay_lines_per_group) - 1) /
                                      np.sqrt(self.num_delay_lines_per_group))

            if self.coupling_matrix_type == CouplingMatrixType.SCALAR:
                # no coupling initialisation
                self.nd_unitary = ND_Unitary()

                # if colorless_feedback_matrix is None:
                # self.alpha = nn.Parameter(
                #     torch.pi * torch.ones(self.num_groups * (self.num_groups - 1) //
                #                 2)) / 4
                self.register_buffer(
                    (torch.pi * torch.ones(self.num_groups * (self.num_groups - 1) // 2)) / 4
                    )                
                # else:
                #     # no coupling allowed - this makes alpha a fixed parameter
                #     self.register_buffer(
                #         "alpha",
                #         torch.zeros(self.num_groups * (self.num_groups - 1) // 2))

            elif self.coupling_matrix_type == CouplingMatrixType.FILTER:
                self.coupling_matrix_order = self.coupling_matrix_order
                # order - 1 unit norm householder vectors
                self.unit_vectors = nn.Parameter(
                    torch.randn(self.num_groups,
                                self.coupling_matrix_order - 1))

                self.unitary_matrix = nn.Parameter(
                    (2 * torch.rand(self.num_groups, self.num_groups) - 1) /
                    np.sqrt(self.num_groups))

                self.fir_paraunitary = FIRParaunitary(
                    self.num_groups, self.coupling_matrix_order)

    def forward(self, z: torch.tensor) -> torch.tensor:
        """Compute (D_m(z^{-1}) - A(z)Gamma(z))^{-1}"""
        num_freq_points = len(z)
        # this will be of size num_freq_points x Ndel x Ndel
        D = torch.diag_embed(torch.unsqueeze(z, dim=-1)**self.delays)

        if self.use_absorption_filters:
            # are the gains in biquads?
            if isinstance(self.delay_line_gains, list):
                # this is of size Ndel x Ndel x num_freq_points
                Gamma = torch.zeros(
                    (self.num_delays, self.num_delays, num_freq_points),
                    dtype=torch.complex64)
                for k in range(self.num_delays):
                    Gamma[k, k, :] = self.delay_line_gains[k](z)
            else:
                # this is of size Ndel x Ndel x num_freq_points
                Gamma = torch.einsum('ij,jk->ijk', torch.eye(self.num_delays),
                                     self.delay_line_gains(z))
        else:
            # this is of size Ndel x Ndel
            Gamma = to_complex(torch.diag(self.delay_line_gains))

        # get coupled feedback matrix of size Ndel x Ndel
        if self.coupling_matrix_type == CouplingMatrixType.RANDOM:
            self.coupled_feedback_matrix = to_complex(
                self.ortho_param(self.random_feedback_matrix))
        else:
            self.coupled_feedback_matrix = self.get_coupled_feedback_matrix()

        # this should be of size num_freq_pts x Ndel x Ndel
        if self.coupling_matrix_type in (CouplingMatrixType.SCALAR,
                                         CouplingMatrixType.RANDOM):
            A = self.coupled_feedback_matrix.unsqueeze(0).repeat(
                num_freq_points, 1, 1)

        elif self.coupling_matrix_type == CouplingMatrixType.FILTER:
            # view converts z into a 2D matrix of size num_freq_pts x 1,
            # but it does not copy the data, hence z ultimately remains unaffected.
            # the exponetiation converts it to order x num_freq_points matrix x order,
            # and permute converts it to order x num_freq_pts

            A = torch.einsum(
                'jim, mn -> jimn', self.coupled_feedback_matrix, (z.view(
                    -1,
                    1)**-torch.arange(0, self.coupling_matrix_order)).permute(
                        1, 0))
            A = torch.sum(A, dim=2).permute(2, 0, 1).to(torch.complex64)

        # A(z) Gamma(z)
        if self.use_absorption_filters:
            # invert diagonal matrix
            Gamma_inv = torch.diag_embed(1.0 / torch.diagonal(Gamma),
                                         dim1=0,
                                         dim2=1)
            Ddecay = D * Gamma_inv.permute(-1, 0, 1)

        else:
            # invert a diagonal matrix
            Gamma_inv = torch.diag(1.0 / torch.diagonal(Gamma))
            Ddecay = D * Gamma_inv.unsqueeze(0).repeat(num_freq_points, 1, 1)

        # the inverse will be taken along the last 2 dimensions
        # the size is num_freq_pts x Ndel x Ndel
        # this is a complex double, but einsum can only handle complex float
        return (torch.linalg.inv(Ddecay - A)).to(torch.complex64)

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
            alpha = self.alpha.clamp(min=-np.pi, max=np.pi)
            phi = self.nd_unitary(alpha, self.num_groups)
            # assert is_unitary(phi)[0]

        elif self.coupling_matrix_type == CouplingMatrixType.FILTER:
            # generate householder matrix from unitary vector
            # ensure that the vectors have unit norm
            unit_vectors = self.unit_vectors / (
                torch.norm(self.unit_vectors, dim=0, keepdim=True) + self._eps)
            phi = self.fir_paraunitary(self.ortho_param(self.unitary_matrix),
                                       unit_vectors)
            # assert is_paraunitary(phi)[0]
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
            try:
                coupled_feedback_matrix = block_M * torch.kron(
                    self.phi, ones_matrix)
            except Exception:
                coupled_feedback_matrix = block_M

        elif self.coupling_matrix_type == CouplingMatrixType.FILTER:
            # computed as A[k] = M_block circ (Phi[k] otimes 1)
            coupled_feedback_matrix = torch.zeros(
                (self.num_delays, self.num_delays, self.coupling_matrix_order))
            for k in range(self.coupling_matrix_order):
                coupled_feedback_matrix[..., k] += block_M * torch.kron(
                    self.phi[..., k], ones_matrix)

        return to_complex(coupled_feedback_matrix)

    def print(self):
        """Print the parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def get_parameters(self) -> Tuple:
        """Return the model parameters as a Tuple"""
        if self.coupling_matrix_type == CouplingMatrixType.RANDOM:
            return self.ortho_param(self.random_feedback_matrix)
        else:
            M = [self.ortho_param(self.M[i]) for i in range(self.num_groups)]
            Phi = self.phi
            L0 = self.ortho_param(self.unitary_matrix)
            unit_vectors = self.unit_vectors / (
                torch.norm(self.unit_vectors, dim=0, keepdim=True) + self._eps)
            coupled_feedback_matrix = self.get_coupled_feedback_matrix()

        delay_line_gains = self.delay_line_gains
        if hasattr(self, 'common_decay_times'):
            return (M, Phi, L0, unit_vectors, coupled_feedback_matrix,
                    delay_line_gains, self.common_decay_times)
        else:
            return (M, Phi, L0, unit_vectors, coupled_feedback_matrix,
                    delay_line_gains)

    @torch.no_grad()
    def get_param_dict(self) -> Dict:
        """Return the model parameters as a dict"""
        param_np = {}
        param_np['delay_line_gains'] = self.delay_line_gains
        if hasattr(self, 'common_decay_times'):
            param_np['common_decay_times'] = self.common_decay_times
        if self.coupling_matrix_type == CouplingMatrixType.RANDOM:
            param_np[
                'coupled_feedback_matrix'] = self.coupled_feedback_matrix.squeeze(
                ).cpu().numpy()
        else:
            param_np['coupling_matrix'] = self.phi.squeeze().cpu().numpy()
            param_np['individual_mixing_matrix'] = self.M.squeeze().cpu(
            ).numpy()
            param_np[
                'coupled_feedback_matrix'] = self.coupled_feedback_matrix.squeeze(
                ).cpu().numpy()
            param_np['unitary_matrix'] = self.unitary_matrix.squeeze().cpu(
            ).numpy()
            param_np['unit_vectors'] = self.unit_vectors.squeeze().cpu().numpy(
            )
        return param_np
