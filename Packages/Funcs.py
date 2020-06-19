import numpy as np
import torch
from math import pi
from Packages.RegistrationFunc import *
from IPython.core.debugger import set_trace


def trKsquare(B, A):  
    # inputs: B, A: two tensors of size 2 x 2 x s x t...
    # output kappa = tr(K^2) of size s x t..., where K = inv(B)A
    
    A, B = A.permute(A.dim()-2, A.dim()-1, *range(A.dim()-2)), B.permute(A.dim()-2, A.dim()-1, *range(A.dim()-2))
    a1,a2=A[0,0],A[1,1]
    b1,b2=B[0,0],B[1,1]
    a3=(A[0,1]+A[1,0])/2.0 
    b3=(B[0,1]+B[1,0])/2.0
    
    ##
    ell1=b1.sqrt()
    ell3=b3/ell1
    ell2=(b2-ell3**2).sqrt()

    w1=a1/ell1**2
    mu=ell3/ell1
    w3=(a3-mu*a1)/(ell1*ell2)
    w2=(a2-2.0*mu*a3+mu**2*a1)/(ell2**2)
    
    ##
    eps1=1e-12
    eps2=1e-12  

    eta=(w1-w2)/(2.0*w3)
    s = torch.where(eta >= 0.0, torch.ones(1,dtype=torch.double), -torch.ones(1,dtype=torch.double))
    tau=s/(eta.abs() + (1.0+eta**2).sqrt())
           
    tau=torch.where((w3.abs()-eps2*(w1-w2).abs())>= 0.0, tau, w3/(w1-w2))
    tau=torch.where((w3.abs()-eps1*(w1*w2).sqrt())>=0.0, tau, torch.zeros(1,dtype=torch.double))
    
    lambda1 = w1+w3*tau
    lambda2 = w2-w3*tau
    kappa=torch.log(lambda1)**2 + torch.log(lambda2)**2
    return kappa


def alb_to_abc(alpha,lambd,beta):
    return alpha, alpha*lambd+beta/16, beta


def abc_to_alb(a,b,c): 
    # maps the 3-parameter constants to constants with respect to the new split metric
    if a==0:
        lambd = 0
    else: 
        lambd = (16*b-c)/(16*a)
    return a, lambd, c


def Squared_distance_abc(gq1, gq2, a, b, c):
    alpha, lambd, beta = abc_to_alb(a,b,c)
    return Squared_distance(gq1, gq2, alpha, lambd, beta)

    
def Squared_distance(gq1, gq2, alpha, lambd, beta):
    
    # calculate the square of the distance with respect to SRNFs
    dist2_q = torch.einsum("imn,imn->mn",[gq1[1] - gq2[1], gq1[1] - gq2[1]])
    
    # calculate the square of the distance with respect to the induced metrics 
    # make g1, g2 of size m x n x 2 x 2
    g1, g2 = gq1[0].permute(2,3,0,1), gq2[0].permute(2,3,0,1)
    
    inv_g1 = torch.zeros(g1.size(), dtype=torch.double) # + 1e-7
    inv_g1[1:g1.size(0)-1] = torch.inverse(g1[1:g1.size(0)-1])
#     set_trace()
    inv_g1_g2 = torch.einsum("...ik,...kj->...ij",[inv_g1, g2])  # mxn*2*2
    
    trK0square = torch.zeros(*g1.shape[:2],dtype=torch.double)
    trK0square[1:g1.size(0)-1]=trKsquare(g1[1:g1.size(0)-1], g2[1:g1.size(0)-1])-(torch.log(torch.det(inv_g1_g2[1:g1.size(0)-1])))**2/2
    
    theta = torch.min((trK0square/lambd + 1e-7).sqrt()/4, torch.tensor([np.pi],dtype=torch.double))
    
    alp, bet = (torch.det(g1) + 1e-7).pow(1/4), (torch.det(g2) + 1e-7).pow(1/4)
    
    dist2_g = 16*lambd*(alp**2 - 2*alp*bet*torch.cos(theta) + bet**2)
    
    if alpha==0:
        return integral_over_s2(beta*dist2_q)
    elif beta==0:
        return integral_over_s2(alpha*dist2_g)
    else:
        return integral_over_s2(alpha*dist2_g + beta*dist2_q)


def f_to_gqf(f):
    m, n = f.shape[-2:]
    PHI, THETA = torch.meshgrid([torch.linspace(0, np.pi, m, dtype=torch.double), 
                                 torch.linspace(0, 2 * np.pi, n, dtype=torch.double)])
    df = torch.zeros(3, 2, m, n, dtype=torch.double)
    Xf10, df[0, 1] = gradient_map(f[0])  # 1/sin(phi)d/dtheta, d/dphi
    Xf20, df[1, 1] = gradient_map(f[1])
    Xf30, df[2, 1] = gradient_map(f[2])

    # adding 1e-7 is for automatic differentiation
    df[0, 0], df[1, 0], df[2, 0] = Xf10/(1e-7 + torch.sin(PHI)), Xf20/(1e-7 + torch.sin(PHI)), Xf30/(
                1e-7 + torch.sin(PHI))
    df[:, 0, [0, m - 1], :] = 0  # when phi is 0 and pi, df/dtheta = 0: singularity
    
    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    Norm_n = (torch.einsum("imn,imn->mn", [n_f, n_f]) + 1e-7).sqrt()

    inv_sqrt_norm_n = 1/ Norm_n.sqrt()
    
    return torch.einsum("kimn,kjmn->ijmn",[df, df]), torch.einsum("imn,mn->imn",[n_f, inv_sqrt_norm_n])


def torch_det(A):  # size(A): k*k*m*n
    return torch.det(A.permute(2,3,0,1))


def torch_inverse(A): # size(A): k*k*m*n
    return torch.inverse(A.permute(2,3,0,1)).permute(2,3,0,1)


# integrate over S2
def integral_over_s2(func):
    # ---input: func: ...k*s*m*n
    # ---output: the integral of func: ...k*s

    m, n = func.shape[-2:]
    phi = torch.linspace(0, pi, m, dtype=torch.double)
    theta = torch.linspace(0, 2 * pi, n, dtype=torch.double)
    PHI, THETA = torch.meshgrid([phi, theta])
    F = func * torch.sin(PHI)
    return torch_trapz_2d(torch.einsum("...ij->ij...", [F]), pi / (m - 1), 2 * pi / (n - 1))


def torch_trapz_2d(func, dphi, dtheta):  # func: m*n*3*2*...
    # trapz function in the first 2D slice
    int_1 = torch.sum(func[0:func.size(0) - 1], 0) + torch.sum(func[1:func.size(0)], 0)
    int_fun = torch.sum(int_1[0:int_1.size(0) - 1], 0) + torch.sum(int_1[1:int_1.size(0)], 0)
    return int_fun * dphi * dtheta / 4


def torch_gradient(F, dtheta, dphi):
    dF_phi, dF_theta = torch.zeros(F.size(), dtype=torch.double), torch.zeros(F.size(), dtype=torch.double)
    dF_theta[:, 0], dF_theta[:, F.size(1) - 1] = (F[:, 1] - F[:, 0]) / dtheta, (
                F[:, F.size(1) - 1] - F[:, F.size(1) - 2]) / dtheta
    dF_theta[:, 1: F.size(1) - 1] = (F[:, 2: F.size(1)] - F[:, 0:F.size(1) - 2]) / (2 * dtheta)

    dF_phi[0], dF_phi[F.size(0) - 1] = (F[1] - F[0]) / dphi, (F[F.size(0) - 1] - F[F.size(0) - 2]) / dphi
    dF_phi[1: F.size(0) - 1] = (F[2: F.size(0)] - F[0:F.size(0) - 2]) / (2 * dphi)
    return dF_theta, dF_phi


def gradient_map(f):
    m, n = f.size()
    dm, dn = pi / (m - 1), 2 * pi / (n - 1)
    F = torch.zeros(m + 2, n + 2, dtype=torch.double)  
    F[1: m + 1, 1: n + 1] = f
    F[0, 1: int((n + 1) / 2) + 1], F[0, int((n + 1) / 2): n + 1] \
        = f[1, int((n + 1) / 2) - 1: n], f[1, 0: int((n + 1) / 2)]
    F[m + 1, 1: int((n + 1) / 2) + 1], F[m + 1, int((n + 1) / 2): n + 1] \
        = f[m - 2, int((n + 1) / 2) - 1: n], f[m - 2, 0: int((n + 1) / 2)]
    F[:, 0], F[:, n + 1] = F[:, n - 1], F[:, 2]
    [X0, Y0] = torch_gradient(F, dn, dm)
    return X0[1:m + 1, 1:n + 1], Y0[1:m + 1, 1:n + 1]


def torch_expm(X):
    ''' 
    input: X = [X1, X2, X3]
    output: the matrix exponential of the anti-symmetric matrix of the form AMX = 
    [  0, X1, X2
     -X1,  0, X3
     -X2, x3,  0]
    '''
    MX = torch.zeros(3,3,dtype=torch.double)
    MX[0, 1], MX[0,2], MX[1,2] = X[0], X[1], X[2]
    AMX = MX - MX.t()
    
    XN = torch.norm(X) + 1e-7
    return torch.eye(3,dtype=torch.double) + torch.sin(XN)*AMX/XN + (1-torch.cos(XN))*AMX@AMX/XN**2
