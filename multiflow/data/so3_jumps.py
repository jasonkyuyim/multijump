import numpy as np
import math
import sys
import scipy
import torch
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
from torch.func import jacrev
from openfold.utils import rigid_utils
from openfold.utils.rigid_utils import convert_to_upper

# def get_uniform_placement_on_3_sphere(r=1, N=500):
#     """
#     Uniformly place points on a 3-sphere following a more accurate discretization. 
#     Inspired from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
#     """
#     Ncount = 0
#     surface_area = (2 * (np.pi ** 2)) * (r ** 3)  # Surface area of a 3-sphere
#     a = surface_area / N  # Surface area per bin
#     d = np.power(a, 1/3)  # Approximate bin size
#     M_theta = round(np.pi / d)  # Number of elements along theta
#     d_theta = np.pi / M_theta  # Step size for theta

#     points = []

#     for m in range(M_theta):
#         theta = np.pi * (m + 0.5) / M_theta  # Theta values (0 to pi)
#         M_phi = round(np.pi * np.sin(theta) / d)  # Number of elements along phi
#         d_phi = np.pi / M_phi  # Step size for phi
        
#         for n in range(M_phi):
#             phi = np.pi * (n + 0.5) / M_phi  # Phi values (0 to pi)
#             M_zeta = round(2 * np.pi * np.sin(theta) * np.sin(phi) / d)  # Number of elements along zeta
#             d_zeta = 2 * np.pi / M_zeta  # Step size for zeta

#             for k in range(M_zeta):
#                 zeta = 2 * np.pi * k / M_zeta  # Zeta values (0 to 2pi)

#                 # Convert spherical coordinates to 4D Cartesian coordinates
#                 u = r * np.sin(theta) * np.sin(phi) * np.sin(zeta)
#                 x = r * np.sin(theta) * np.sin(phi) * np.cos(zeta)
#                 y = r * np.sin(theta) * np.cos(phi)
#                 z = r * np.cos(theta)

#                 points.append((u, x, y, z))
#                 Ncount += 1
#     grid = torch.from_numpy(np.stack(points))
#     grid = grid/torch.norm(grid,dim=1)[:,None] # unnecessary (just to make sure)
#     return grid

# def get_uniform_quaternion_bins(r=1, N=500):
#     """
#     Uniformly distribute points on SO(3) using quaternions.
#     This function generates N points on the 3-sphere S^3 that correspond to rotations in SO(3),
#     ensuring that each bin has equal volume.
#     """
#     sphere_grid = get_uniform_placement_on_3_sphere(r=r,N=2*N)
#     half_volume_mask = sphere_grid[:,0]==0
#     return sphere_grid[sphere_grid[:,0]>=0], half_volume_mask

# def gamma(t, x, y, eps=0.001):
#     """Spherical Linear Interpolation between two points on the 2-sphere"""

#     # Normalize the points to ensure they lie on the 3-sphere
#     x = x / torch.norm(x)
#     y = y / torch.norm(y) #, dim=1)[:,None]

#     # Compute the angle between the two points
#     dot_product = eps + (1-2*eps) * torch.dot(x,y)
#     omega = torch.arccos(dot_product)
#     sin_omega = torch.sin(omega)
#     p = (torch.sin((1 - t) * omega) / sin_omega) * x + (torch.sin(t * omega) / sin_omega) * y

#     return p

# def batch_gamma(t, x, y):
#     """Spherical Linear Interpolation between two points on the 2-sphere"""

#     # Normalize the points to ensure they lie on the 3-sphere
#     x = x / torch.norm(x,dim=-1)[..., None]
#     y = y / torch.norm(y,dim=-1)[..., None] #, dim=1)[:,None]

#     # Compute the angle between the two points
#     omega = torch.arccos((x*y).sum(dim=-1))

#     # Compute the interpolated point
#     sin_omega = torch.clip(torch.sin(omega),min=0.00001)
#     p = (torch.sin((1 - t) * omega) / sin_omega)[...,None] * x + (torch.sin(t * omega) / sin_omega)[...,None] * y

#     return p
    
# def psi(t,x,y):
#     return torch.norm(x) * gamma(t,x,y)

# def inv_psi(t, x, y):
#     return torch.norm(x) * gamma(1/torch.clip(1-t,min=0.0001), y, x)

# def compute_det_inv_psi_t(t, x1, x0):
#     jac_map = (lambda t,x,y: jacrev(lambda x_1: inv_psi(t,x_1,y))(x))
#     comp_jacobian = torch.func.vmap(jac_map, in_dims=(0))(t,x0,x1)
#     det_jac_inv_gamma = torch.linalg.det(comp_jacobian)
#     return det_jac_inv_gamma

# def compute_dt_det_inv_gamma_t(t, x, x1, inverse_method=False, scale_factor=1.0):
#     if not inverse_method:
#         t = deepcopy(t).double()
#         t.requires_grad_(True)
#         abs_det_inv_gamma_t = torch.abs(compute_det_inv_psi_t(t, x1.double(), x.double()))
#         abs_det_inv_gamma = abs_det_inv_gamma_t/scale_factor
#         abs_det_inv_gamma_t.sum().backward()
#         return scale_factor * abs_det_inv_gamma_t.detach().float(), scale_factor*t.grad.float()

# def compute_pt(t, x1, x, eps=0.001):
#     dot_product = (x1 * x).sum(dim=1)
#     omega = torch.arccos(dot_product)
#     support_mask = (omega < torch.pi * (1-t)-eps).float()
#     return support_mask*torch.abs(compute_det_inv_psi_t(t,x1,x).nan_to_num(nan=0.0))/(2*torch.pi)

# def sphere_condot_lambda_t(t, x, x1, eps=0.001, out_support_val=50):
#     dot_product = (x1 * x).sum(dim=1)
#     omega = torch.arccos(dot_product)
#     support_mask = (omega < torch.pi * (1-t)-eps)
#     det_inv_gamma_t, dt_det_inv_gamma_t = compute_dt_det_inv_gamma_t(t, x, x1)
#     return (support_mask*torch.relu(-dt_det_inv_gamma_t)/det_inv_gamma_t)+((~support_mask)*out_support_val)

# def compute_unnormalized_J_t(t, x, x1, eps=0.0001, out_of_support_val=0.0):
#     dot_product = (x1 * x).sum(dim=1)
#     omega = torch.arccos(dot_product)
#     support_mask = (omega < torch.pi * (1-t)-eps)
#     det_inv_gamma_t, dt_det_inv_gamma_t = compute_dt_det_inv_gamma_t(t, x, x1)
#     return support_mask * torch.relu(dt_det_inv_gamma_t)+(~support_mask)*out_of_support_val

# def compute_bin_normalized_J_t(t, uniform_grid, x1, eps=0.0001, out_of_support_val=0.0):
#     len_unif_grid = uniform_grid.shape[0]
#     uniform_grid = torch.repeat_interleave(uniform_grid, len(x1), dim=0)
#     x1 = x1.repeat(len_unif_grid, 1)
#     dot_product = (x1 * uniform_grid).sum(dim=1)
#     omega = torch.arccos(dot_product)
#     t = t.repeat(len_unif_grid)
#     support_mask = (omega < torch.pi * (1-t)-eps)
#     det_inv_gamma_t, dt_det_inv_gamma_t = compute_dt_det_inv_gamma_t(t, uniform_grid, x1)
#     unnormalized_J_t = support_mask * torch.relu(dt_det_inv_gamma_t)+(~support_mask) * out_of_support_val
#     unnormalized_J_t = unnormalized_J_t.reshape(len_unif_grid,-1)
#     normalized_J_t = unnormalized_J_t/torch.clip(unnormalized_J_t.sum(dim=0)[None,:],min=1e-10)
#     return normalized_J_t.transpose(1,0)


# def convert_to_upper_sphere(x):
#     """Function to convert elements to the upper hemisphere."""
#     if x.dim() == 1:
#         return (x * (x[-1]>0).detach() + (-x) * (x[-1]<=0).detach())
#     elif x.dim() == 2:
#         return (x * (x[:,-1]>0)[:,None].detach() + (-x) * (x[:,-1]<=0)[:,None].detach())
#     elif x.dim() == 3:
#         return (x * (x[:,:,-1]>0)[:,:,None].detach() + (-x) * (x[:,:,-1]<=0)[:,:,None].detach())
#     else:
#         sys.exit("Unknown number of dimensions: ", x.dim())

def get_uniform_placement_on_3_sphere(r=1, N=500):
    """
    Uniformly place points on a 3-sphere following a more accurate discretization. 
    Inspired from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    Ncount = 0
    surface_area = (2 * (np.pi ** 2)) * (r ** 3)  # Surface area of a 3-sphere
    a = surface_area / N  # Surface area per bin
    d = np.power(a, 1/3)  # Approximate bin size
    M_theta = round(np.pi / d)  # Number of elements along theta
    d_theta = np.pi / M_theta  # Step size for theta

    points = []

    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta  # Theta values (0 to pi)
        M_phi = round(np.pi * np.sin(theta) / d)  # Number of elements along phi
        d_phi = np.pi / M_phi  # Step size for phi
        
        for n in range(M_phi):
            phi = np.pi * (n + 0.5) / M_phi  # Phi values (0 to pi)
            M_zeta = round(2 * np.pi * np.sin(theta) * np.sin(phi) / d)  # Number of elements along zeta
            d_zeta = 2 * np.pi / M_zeta  # Step size for zeta

            for k in range(M_zeta):
                zeta = 2 * np.pi * k / M_zeta  # Zeta values (0 to 2pi)

                # Convert spherical coordinates to 4D Cartesian coordinates
                u = r * np.sin(theta) * np.sin(phi) * np.sin(zeta)
                x = r * np.sin(theta) * np.sin(phi) * np.cos(zeta)
                y = r * np.sin(theta) * np.cos(phi)
                z = r * np.cos(theta)

                points.append((u, x, y, z))
                Ncount += 1
    grid = torch.from_numpy(np.stack(points))
    grid = grid/torch.norm(grid,dim=1)[:,None] # unnecessary (just to make sure)
    return grid

def get_uniform_quaternion_bins(r=1, N=500):
    """
    Uniformly distribute points on SO(3) using quaternions.
    This function generates N points on the 3-sphere S^3 that correspond to rotations in SO(3),
    ensuring that each bin has equal volume.
    """
    sphere_grid = get_uniform_placement_on_3_sphere(r=r,N=2*N)
    half_volume_mask = sphere_grid[:,0]==0
    return sphere_grid[sphere_grid[:,0]>=0], half_volume_mask

class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z),
        )


class LogIveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.log(torch.tensor(output).to(z.device))

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * ((-1.5 + torch.sqrt( (self.v+0.5) ** 2 + z ** 2) + torch.sqrt( (self.v+1) ** 2 + z ** 2) ) / (2*z) -1),
        )
        # grad_output * ((-1 + torch.sqrt( (self.v+1) ** 2 + z ** 2) ) / z -1)
        # grad_output * ((-0.5 + torch.sqrt( (self.v+0.5) ** 2 + z ** 2) ) / z -1)
class Ive(torch.nn.Module):
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply
logive = LogIveFunction.apply

# source: https://arxiv.org/pdf/1606.02008.pdf
def ive_fraction_approx(v, z):
    # I_(v/2)(k) / I_(v/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
    return z / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(z, 2), 0.5))


# source: https://arxiv.org/pdf/1902.02603.pdf
def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(eps))
        )

    delta_0 = delta_a(0.0)
    delta_2 = delta_a(2.0)
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(eps)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(eps)
    )

    return (B_0 + B_2) / 2.0
    
class VonMisesFisher(torch.distributions.Distribution):
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # option 1:
        return self.loc * (
            ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        )
        # option 2:
        # return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # options 3:
        # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, size):
        with torch.no_grad():
            return self.rsample(torch.Size((size[0],)))

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )

        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = accept_idx < 0
            accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        # option 1:
        output = (
            -self.scale
            * ive(self.__m / 2, self.scale)
            / ive((self.__m / 2) - 1, self.scale)
        )
        # option 2:
        # output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # option 3:
        # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        #print('scale:',self.scale)

        #print(f'output shape ok: {output.shape}')
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
        )

        #print(f'output shape not ok: {output.shape}')
        return output #.view(*(output.shape[:-1]))


class VMFPath:    
    def __init__(
            self,
            *,
            path_name='vmf',
            kappa_max: float = 100.0,
            kappa_min: float = 0.01,
            kappa_alpha: float = 4.0,
            t_min: float = 0.01,
            logive_approx=False,
            upper_half=True
        ):
        self.path_name = path_name
        self.kappa_max = kappa_max
        self.logive_approx = logive_approx
        self.upper_half = upper_half
        self.kappa_min = kappa_min
        self.kappa_alpha = kappa_alpha
        self.t_min = t_min
        
    def sample_noisy(self, x1, t):
        kappa_t = self.kappa(t.view(-1,1))
        xt = VonMisesFisher(loc=x1, scale=kappa_t).rsample(torch.Size((1,)))
        return rigid_utils.convert_to_upper(xt.squeeze())
    
    def exp_const_term(self, x1, xt, t):
        kappa_t = self.kappa(t.view(-1,1)).squeeze()
        amb_D = x1.shape[1]
        exp_term = kappa_t * (x1 * xt).sum(dim=1)
        if self.logive_approx:
            const_term = (amb_D / 2 - 1) * torch.log(kappa_t) - (amb_D / 2) * math.log(2 * math.pi) - (kappa_t + logive(amb_D / 2 - 1, kappa_t))
        else:
            const_term = (amb_D / 2 - 1) * torch.log(kappa_t) - (amb_D / 2) * math.log(2 * math.pi) - (kappa_t + torch.log(ive(amb_D / 2 - 1, kappa_t)))
        return exp_term, const_term
    
    def log_p_t(self, x1, xt, t):
        if self.upper_half:
            exp_term_pos, const_term_pos = self.exp_const_term(x1, xt, t)
            exp_term_neg, const_term_neg = self.exp_const_term(x1, -xt, t)
            log_prob_pos = const_term_pos + exp_term_pos
            log_prob_neg = const_term_neg + exp_term_neg 
            prob_pos = torch.exp(log_prob_pos)
            prob_neg = torch.exp(log_prob_neg)
            return torch.log(prob_pos + prob_neg)
        else:
            exp_term, const_term = self.exp_const_term(x1, xt, t)
            log_prob = const_term + exp_term
            return log_prob
    
    def p_t(self, x1, xt, t):
        if self.upper_half:
            exp_term_pos, const_term_pos = self.exp_const_term(x1, xt, t)
            exp_term_neg, const_term_neg = self.exp_const_term(x1, -xt, t)
            log_prob_pos = const_term_pos.flatten() + exp_term_pos.flatten()  
            log_prob_neg = const_term_neg.flatten() + exp_term_neg.flatten()  
            prob_pos = torch.exp(log_prob_pos)
            prob_neg = torch.exp(log_prob_neg)
            return prob_pos + prob_neg
        else:
            exp_term, const_term = self.exp_const_term(x1, xt, t)
            log_prob = const_term.flatten() + exp_term.flatten()  
            return torch.exp(log_prob)
            
    def kappa(self, t, eps: float = 0.001):
        if self.path_name == 'vmf':
            return self.kappa_min + (self.kappa_max - self.kappa_min + 1.) **(t*self.kappa_alpha) - 1.
        elif self.path_name == 'log_t':
            return self.kappa_min + torch.clip(-torch.log(torch.clip(1-t,min=eps)),min=eps, max=self.kappa_max)
        elif self.path_name == "monomial":
            return self.kappa_min+(self.kappa_max-self.kappa_min) * (t**self.kappa_alpha)
        else:
            raise NotImplementedError

    def lambda_t(self, x1, xt, t):
        # t = deepcopy(t).double()
        t.requires_grad_(True)
        log_p_t = self.log_p_t(x1, xt, t)
        output = log_p_t.sum()
        output.sum().backward()
        dt_log_pt = t.grad.float().detach()
        lambda_t_val = torch.relu(-dt_log_pt)
        return lambda_t_val

    def get_bow_grid(self, n_grid_points: int = 100):
        if self.upper_half:
            angle_grid = torch.linspace(0,np.pi/2,n_grid_points)
        else:
            angle_grid = torch.linspace(-np.pi/2,np.pi/2,n_grid_points)
        bow_grid = torch.stack([torch.cos(angle_grid),torch.sin(angle_grid),torch.zeros_like(angle_grid),torch.zeros_like(angle_grid)],dim=1)
        return bow_grid


    def compute_cashed_dt_pt(self, x1, n_time_points: int = 1000, n_grid_points: int = 100, device: str = 'cpu'):
        self.bow_grid = self.get_bow_grid(n_grid_points=n_grid_points)
        self.dt_p_t_time_cach = torch.linspace(0.0,1,n_time_points)
        x1 = torch.tensor([[1.0,0.0,0.0,0.0]], requires_grad=True)
        dt_p_t_cach_list = []
        for idx, t in enumerate(self.dt_p_t_time_cach):
            t_vec = torch.ones(size=(1,)) * t
            with torch.enable_grad():
                dt_p_t = self.compute_dt_pt(x1.to(device), self.bow_grid.to(device), t_vec.to(device))
            dt_p_t_cach_list.append(dt_p_t.cpu().detach())

        dot_product = (self.bow_grid.to(device) * x1.squeeze()[None,:].to(device)).sum(dim=1)
        if self.upper_half:
            omega_pos = torch.arccos(dot_product)
            omega_neg = torch.arccos(-dot_product)
            omega_dist = torch.min(omega_pos, omega_neg)
        else:
            omega_dist = torch.arccos(dot_product)
        self.omega_dist_cach = omega_dist.cpu().detach()
        self.dt_p_t_cach = torch.stack(dt_p_t_cach_list).cpu().detach().squeeze()
        assert self.dt_p_t_time_cach.shape[0] == self.dt_p_t_cach.shape[0]
        assert self.omega_dist_cach.shape[0] == self.dt_p_t_cach.shape[1]
        self.dt_p_t_cach[0,:] = 0.0
        self.interpolator = RegularGridInterpolator((np.array(self.dt_p_t_time_cach), np.array(self.omega_dist_cach)), np.array(self.dt_p_t_cach))

    def compute_dt_pt(self, x1, uniform_grid, t):
        # Reshape tensors for broadcasting
        n = x1.shape[0]
        m = uniform_grid.shape[0]
        d = x1.shape[1]
        
        x1_reshaped = x1.unsqueeze(1)  # shape (n, 1, d)
        uniform_grid_reshaped = uniform_grid.unsqueeze(0)  # shape (1, m, d)
        if t.dim() == 1:
            t_reshaped = t.unsqueeze(1)
        else:
            t_reshaped = t
        # Broadcasting to compute all combinations (both tensors will broadcast to (n, m, d))
        x1_combinations = x1_reshaped.expand(n, m, d).reshape(n*m,d)  # shape (n, m, d)
        uniform_grid_combinations = uniform_grid_reshaped.expand(n, m, d).reshape(n*m,d)  # shape (n, m, d)
        t_combinations = t_reshaped.expand(n,m).reshape(n*m)
        
        t_combinations = deepcopy(t_combinations).double()
        t_combinations.requires_grad_(True)
        p_t = self.p_t(x1_combinations, uniform_grid_combinations, t_combinations)
        output = p_t.sum()
        output.sum().backward()
        dt_pt = t_combinations.grad.float().detach()
        dt_pt = dt_pt.reshape(n,m)
        return dt_pt

    def cached_J_t(self, x1, uniform_grid, t, temp):
        n_bins = len(uniform_grid)
        dot_product = (uniform_grid[None,:,:] * x1[:,None,:]).sum(dim=2)
        if self.upper_half:
            omega_pos = torch.arccos(dot_product)
            omega_neg = torch.arccos(-dot_product)
            omega_dist = torch.min(omega_pos, omega_neg)
        else:
            omega_dist = torch.arccos(dot_product)
        t = t.reshape(-1,1)
        t_expand = t.expand(t.shape[0], omega_dist.shape[1])
        t_flatten = torch.clip(t_expand.flatten(), min=float(self.dt_p_t_time_cach.min()), max=float(self.dt_p_t_time_cach.max()))
        omega_flatten = torch.clip(omega_dist.flatten(), min=float(self.omega_dist_cach.min()), max=float(self.omega_dist_cach.max()))
        points = torch.stack([t_flatten, omega_flatten],dim=0).transpose(1,0).detach().cpu().numpy()
        interpolated_values = self.interpolator(points)
        dt_p_t_vals = torch.tensor(interpolated_values).reshape(-1,n_bins).to(x1.device)
        unnormalized_J_t = torch.nn.functional.relu(dt_p_t_vals)
        J_t_dist = unnormalized_J_t/(unnormalized_J_t.sum(dim=1) + 1e-10)[:,None]
        bool_mask = (J_t_dist.sum(dim=1) < 0.99)
        if bool_mask.any():
            # Reshape tensors for broadcasting
            x1 = x1[bool_mask]
            t = t[bool_mask]
            n = x1.shape[0]
            m = uniform_grid.shape[0]
            d = x1.shape[1]
            
            x1_reshaped = x1.unsqueeze(1)  # shape (n, 1, d)
            uniform_grid_reshaped = uniform_grid.unsqueeze(0).to(x1.device)  # shape (1, m, d)
            if t.dim() == 1:
                t_reshaped = t.unsqueeze(1)
            else:
                t_reshaped = t
            # Broadcasting to compute all combinations (both tensors will broadcast to (n, m, d))
            x1_combinations = x1_reshaped.expand(n, m, d).reshape(n*m,d)  # shape (n, m, d)
            uniform_grid_combinations = uniform_grid_reshaped.expand(n, m, d).reshape(n*m,d)  # shape (n, m, d)
            t_combinations = t_reshaped.expand(n,m).reshape(n*m).double()
            t_combinations.requires_grad_(True)
            p_t = self.p_t(x1_combinations, uniform_grid_combinations, t_combinations)

            p_t = p_t.reshape(n, m).to(x1.device)**(1/temp)
            j_t_hack = p_t / p_t.sum(dim=1)[:, None]
            J_t_dist[bool_mask] = j_t_hack
        return J_t_dist
        # return mask * j_t_hack + J_t_dist * (1 - mask)

def convert_x1_to_x_t_relative_frame(x_t, x_1):
    batch_size, n_acc, dim_ = x_t.shape
    assert dim_ == 4
    x_t_rot = rigid_utils.quat_to_rot(x_t)
    x_1_rot = rigid_utils.quat_to_rot(x_1)
    x_1_rot_rel = torch.matmul(x_1_rot,x_t_rot.transpose(3,2))
    x_1_rel = rigid_utils.rot_to_quat(x_1_rot_rel)
    return rigid_utils.convert_to_upper(x_1_rel)

def convert_relative_jump_to_absolute(x_t, jump_samples):
    batch_size, n_acc, dim_ = x_t.shape
    assert dim_ == 4
    x_t_rot = rigid_utils.quat_to_rot(x_t)
    jump_samples_rot = rigid_utils.quat_to_rot(jump_samples)
    jump_samples_rot_abs = torch.matmul(jump_samples_rot,x_t_rot)
    jump_samples_abs = rigid_utils.rot_to_quat(jump_samples_rot_abs)
    return rigid_utils.convert_to_upper(jump_samples_abs)

def bin_elbo_loss(
        log_lambda_t: torch.Tensor,
        log_J_t: torch.Tensor,
        time: torch.Tensor,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        bins: torch.Tensor,
        vmf_path,
        use_cache: bool = False,
        reduce: bool = False,
        out_support_val: float = 0.0,
        use_mse: bool = False,
        train_x_t_rel_frame: bool = True,
    ):

    # Reshape values:
    # x_t = convert_to_upper_sphere(x_t)
    # x1 = convert_to_upper_sphere(x1)
    if train_x_t_rel_frame:
        x1 = convert_x1_to_x_t_relative_frame(x_t, x1)
        x_t_rel = convert_x1_to_x_t_relative_frame(x_t, x_t)
        
    batch_size, n_acc = log_lambda_t.shape
    time = time.unsqueeze(1).expand(batch_size, n_acc).reshape(batch_size * n_acc)
    log_lambda_t = log_lambda_t.reshape(batch_size*n_acc)
    log_J_t = log_J_t.reshape(batch_size*n_acc, log_J_t.shape[2])
    x1 = x1.reshape(batch_size*n_acc, 4)
    x_t = x_t.reshape(batch_size*n_acc, 4)
    if train_x_t_rel_frame:
        x_t_rel = x_t_rel.reshape(batch_size*n_acc, 4)

    lambda_t_ref = vmf_path.lambda_t(x1, x_t_rel, time).to(x1.device)
    J_t_ref = vmf_path.cached_J_t(x1, bins, time).to(x1.device)
    Q_t_ref = lambda_t_ref[:,None] * J_t_ref

    # DEBUG: REMOVE FIXING LAMBDA
    #log_lambda_t_ref = torch.log(torch.clip(lambda_t_ref, min=1e-8))
    #log_Q_t = log_lambda_t_ref[:,None] + log_J_t #torch.log(torch.clip(J_t_ref, min=1e-8)).detach()
    log_Q_t = log_lambda_t[:,None] + log_J_t #torch.log(torch.clip(J_t_ref, min=1e-8)).detach()
    Q_t = torch.clip(torch.exp(log_Q_t), min=0.0, max=1e4)

    # Debug MSE loss:
    #lambda_loss = (torch.exp(log_lambda_t) - lambda_t_ref * log_lambda_t).reshape(batch_size, n_acc).mean(dim=1)
    #return lambda_loss
    #((lambda_t_ref-torch.exp(log_lambda_t))**2).reshape(batch_size, n_acc).mean(dim=1)

    # 1. Compute correct bins for X_1:
    if train_x_t_rel_frame:
        sim_matrix = (x_t_rel @ bins.transpose(1,0)) # Take bin closest to relative position
    else:
        sim_matrix = (x_t @ bins.transpose(1,0))
    omega_pos = torch.arccos(sim_matrix)
    omega_neg = torch.arccos(-sim_matrix)
    dist_matrix = torch.min(omega_pos, omega_neg)
    min_mask = (dist_matrix == (dist_matrix.min(dim=1).values[:,None]))
    loss_stay_at_position = (Q_t * (~min_mask).float()).sum(dim=1)
    loss_jump = -(Q_t_ref * log_Q_t * (~min_mask).float()).sum(dim=1)
    loss_per_aa = torch.clip(loss_stay_at_position + loss_jump, min=-1e2, max=1e4)

    n_nan_elements = torch.isnan(loss_per_aa).sum()
    if n_nan_elements > 0:
        print("Replacing # NaN elements with 0.0: ", n_nan_elements)
        loss_per_aa = torch.nan_to_num_(loss_per_aa)

    loss_per_element = loss_per_aa.reshape(batch_size, n_acc).mean(dim=1)
    if reduce:
        loss_reduced = loss_per_element.mean()
        return loss_reduced
    else:
        return loss_per_element


def cross_entropy_loss(
        log_p_1_t: torch.Tensor,
        time: torch.Tensor,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        bins: torch.Tensor,
        vmf_path,
        reduce: bool = False,
    ):

    # Convert everything to frames relative to x_t:
    x1_rel = convert_x1_to_x_t_relative_frame(x_t, x1)

    # Reshape elements (flatten across amino-acid dimension):
    batch_size, n_acc, _ = log_p_1_t.shape    
    time = time.unsqueeze(1).expand(batch_size, n_acc).reshape(batch_size * n_acc)
    log_p_1_t = log_p_1_t.reshape(batch_size*n_acc, log_p_1_t.shape[2])
    x1_rel = x1_rel.reshape(batch_size*n_acc, 4)

    # Similarity matrix
    sim_matrix = (x1_rel @ bins.transpose(1,0)) # Take bin closest to relative position

    # Distance:
    omega_pos = torch.arccos(sim_matrix)
    omega_neg = torch.arccos(-sim_matrix)
    dist_matrix = torch.min(omega_pos, omega_neg)
    min_mask = (dist_matrix == (dist_matrix.min(dim=1).values[:,None]))

    # Cross-entropy loss:
    loss_per_aa = -(log_p_1_t * (min_mask).float()).sum(dim=1)
    n_nan_elements = torch.isnan(loss_per_aa).sum()
    if n_nan_elements > 0:
        print("Replacing # NaN elements with 0.0: ", n_nan_elements)
        loss_per_aa = torch.nan_to_num_(loss_per_aa)

    loss_per_element = loss_per_aa.reshape(batch_size, n_acc).mean(dim=1)
    if reduce:
        loss_reduced = loss_per_element.mean()
        return loss_reduced
    else:
        return loss_per_element

