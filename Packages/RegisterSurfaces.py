import scipy.io as sio
import scipy.optimize as optimize
from Packages.RegistrationFunc import *
from Packages.Funcs import *


# the main program for surface registration
# inputs: f1, f2...two surfaces of size 3*num_phi*num_theta, where phi denotes the polar angle and theta denotes the azimuthal angle.
#                  (To use the code, num_theta must be odd !!!)
#         MaxDegVecFS2...the maximal degree of spherical harmonics for the basis of vector fields on S2
#         a,b,c...weights of the general elastic metric:
#    		    a weights the term that measures the change in metric, 
#    		    b weights the term that measures the change in volume density and 
#    		    c weights the term that measures the change in normal direction
#         maxite...the maximum number of iterations for the whole optimization process and the maximal number of iterations for each optimization 

def opt_overDiff(f1, f2, a, b, c, MaxDegVecFS2, maxiter = (1,50)):
    
    idty = get_idty_S2(*f1.shape[-2:])
    
    # load the basis for tangent vector fields on S2
    mat_vecF = sio.loadmat('Bases/basis_vecFieldsS2_deg25_{0}_{1}.mat'.format(*f1.shape[-2:]))
    
    N_basis_vec = (MaxDegVecFS2 + 1) ** 2 - 1  # half the number of basis for the vector fields on S2
    Basis0_vec = torch.from_numpy(mat_vecF['Basis'])[: N_basis_vec].double()
    Basis_vecFields = torch.cat((Basis0_vec[:, 0], Basis0_vec[:, 1]))  # get a basis of the tangent fields on S2
    
    f = f1
    N_ite, Max_ite_in = maxiter
    
    Dis = []
    for i in range(N_ite):
        X = torch.zeros(Basis_vecFields.size(0), dtype=torch.double) +1e-2
    
        def Opts(X):
            X = torch.from_numpy(X).double().requires_grad_()
            y = opt_dis(X, f, f2, a, b, c, idty, Basis_vecFields)
            
            y.backward()
            return np.double(y.data.numpy()), np.double(X.grad.data.numpy())
    
        def printx(x):
            Dis.append(Opts(x)[0])
    
        res_reg = optimize.minimize(Opts, X, method='BFGS',
                                    jac=True, callback=printx, options={'gtol': 1e-02, 'maxiter': Max_ite_in, 'disp': False})  # True

        gamma = idty + torch.einsum("i, ijkl->jkl", [torch.from_numpy(res_reg.x).double(), Basis_vecFields])
        gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
        
        # get the spherical coordinate representation
        gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))
        
        f = compose_gamma(f, gammaSph)
        
    return f, Dis


# define the functional to be optimized
def opt_dis(X, f1, f2, a, b, c, idty, Basis_vecFields):
    
    gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2

    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
    f1_gamma = compose_gamma(f1, gammaSph)

    gqf1_gamma = f_to_gqf(f1_gamma)
    gqf2 = f_to_gqf(f2)
    
    return Squared_distance_abc(gqf1_gamma, gqf2, a, b, c)



#------------------------
# get an initial guess for the optimal reparametrization

def initialize_over_paraSO3(f1, f2, a, b, c):
    
    idty = get_idty_S2(*f1.shape[-2:])
    
    # load the elements in the icosahedral group
    XIco_mat = sio.loadmat('Bases/skewIcosahedral.mat')
    XIco = torch.from_numpy(XIco_mat['X']).double()
    
    gq2 = f_to_gqf(f2)
    
    # initialize over the icosahedral group
    EIco = torch.zeros(60,dtype=torch.double)
    f1_gammaIco = torch.zeros(60, *f1.size(),dtype=torch.double)
    for i in range(60):
        RIco = torch.einsum("ij,jmn->imn", [torch_expm(XIco[i]), idty])
        gammaIco = torch.stack((Cartesian_to_spherical(RIco[0], RIco[1], RIco[2])))
        f1_gammaIco[i] = compose_gamma(f1, gammaIco)
        gq1_gammaIco = f_to_gqf(f1_gammaIco[i])
        
        EIco[i] = Squared_distance_abc(gq1_gammaIco, gq2, a, b, c)

    # get the index of the smallest value
    Ind = np.argmin(EIco)
    
    X = XIco[Ind] 
    
    # initialize over the group of diffeomorphisms of rotations
    ESO3 = []
    def opt(X):
        X = torch.from_numpy(X).double().requires_grad_()
        R = torch.einsum("ij,jmn->imn", [torch_expm(X), idty])
        
        gamma = torch.stack((Cartesian_to_spherical(R[0] + 1e-7, R[1], (1 - 1e-7) * R[2])))
        f1_gamma = compose_gamma(f1, gamma)
        
        gq1_gamma = f_to_gqf(f1_gamma)
        
        y = Squared_distance_abc(gq1_gamma, gq2, a, b, c)
        y.backward()
        return np.double(y.data.numpy()), np.double(X.grad.data.numpy())
    
    def printx(x):
        ESO3.append(opt(x)[0])
    
    res = optimize.minimize(opt, X, method='BFGS',
                                jac=True, callback=printx, options={'gtol': 1e-02, 'disp': False})  # True

    X_opt = torch.from_numpy(res.x).double()
    R_opt = torch.einsum("ij,jmn->imn", [torch_expm(X_opt), idty])
    gamma_opt = torch.stack((Cartesian_to_spherical(R_opt[0], R_opt[1],R_opt[2])))
    f1_gamma = compose_gamma(f1, gamma_opt)
    return  f1_gamma, ESO3, f1_gammaIco[Ind], EIco

