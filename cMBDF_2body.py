import numpy as np
import numba as nb
from numpy import einsum
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len

root2,ipi=2**0.5,np.pi*1j
half_rootpi=(np.pi**0.5)/2

@nb.jit(nopython=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 + 
                        t * ( 0.09678418 + 
                        t * (-0.18628806 + 
                        t * ( 0.27886807 + 
                        t * (-1.13520398 + 
                        t * ( 1.48851587 + 
                        t * (-0.82215223 + 
                        t * ( 0.17087277))))))))))
    return ans

@nb.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    #if degree == 0:
    #    return 1
    if degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2
    elif degree == 5:
        x1 = (a*x)**5
        x2 = (a*x)**3
        return 32*x1 - 160*x2 + 120*(a*x)


@nb.jit(nopython=True)
def fcut(Rij, rcut): #checked
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)

@nb.jit(nopython=True)
def fcut_with_grad(Rij, rcut):
    arg = (np.pi*Rij)/rcut
    return 0.5*(np.cos(arg)+1), (-np.pi*np.sin(arg))/(2*rcut)

@nb.jit(nopython=True)
def generate_data_with_gradients(size,charges,coods,rconvs_arr,cutoff_r=12.0,n_atm = 2.0):
    rconvs, drconvs = rconvs_arr[0], rconvs_arr[1]
    m, n = rconvs.shape[0], rconvs.shape[1]
    nrs = m*n
    rstep = cutoff_r/rconvs.shape[-1]
    twob = np.zeros((size,size,nrs))
    twob_temps = np.zeros((size,size,8))
    
    #threeb = [np.zeros((size, size, size)) for i in range(6)]

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=np.linalg.norm(rij)

            if rij_norm!=0 and rij_norm<cutoff_r:
                grad_dist = rij/rij_norm
                z2 = charges[j]
                fcutij, gfcut = fcut_with_grad(rij_norm, cutoff_r)
                ind = rij_norm/rstep
                ac = np.sqrt(z*z2)
                pref = ac*fcutij
                #print(ac, fcutij, gfcut)
                twob_temps[i][j][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[j][i][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[i][j][5:] = grad_dist
                twob_temps[j][i][5:] = -grad_dist             
                ind = int(ind)
                id2 = 0
                for i1 in range(m):
                    for i2 in range(n):
                        conv = pref*rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2+=1

    twob_grad = np.zeros((size,nrs,size,3))

    for i in range(size):
        #grad = np.zeros((nrs,size,3))
        grad_temp = np.zeros((nrs,3))

        for j in range(size):
            
            if j!=i:
                
                ac, fcutij, gfcut, rij_norm, ind = twob_temps[i][j][:5]
                grad_dist = -twob_temps[i][j][5:]
                if ac!=0:
                    ind = int(ind)

                    gradfcut = gfcut*grad_dist

                    pref = ac*fcutij
                    id2 = 0

                    for i1 in range(m):
                        for i2 in range(n):
                            grad1 = pref*drconvs[i1][i2][ind]*grad_dist
                            grad2 = (twob[i][j][id2]*gradfcut)/fcutij
                            twob_grad[i][id2][j] = grad1+grad2

                            grad_temp[id2] += -(grad1+grad2)
                            id2+=1

        twob_grad[i,:,i,:] = grad_temp
   
    return twob, twob_grad


@nb.jit(nopython=True)
def generate_data(size,charges,coods,rconvs,cutoff_r=12.0,n_atm = 2.0):

    m, n = rconvs.shape[0], rconvs.shape[1]
    nrs = m*n
    rstep = cutoff_r/rconvs.shape[-1]
    twob = np.zeros((size,size,nrs))
    #threeb = [np.zeros((size, size, size)) for i in range(6)]

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=np.linalg.norm(rij)

            if rij_norm!=0 and rij_norm<cutoff_r:
                z2 = charges[j]
                fcutij = fcut(rij_norm, cutoff_r)
                ind = int(rij_norm/rstep)
                #twob[i][j] = rij_norm,np.sqrt(z*z2),fcutij
                #twob[j][i] = rij_norm,np.sqrt(z*z2),fcutij
                pref = np.sqrt(z*z2)*fcutij
                
                id2 = 0
                for i1 in range(m):
                    for i2 in range(n):
                        conv = pref*rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2+=1

                #for k in range(j+1, size):
                #    rik=atom-coods[k]
                #    rik_norm=np.linalg.norm(rik)
#
                #    if rik_norm!=0 and rik_norm<cutoff_r:
                #        z3=charges[k]**0.8
                #        
                #        rkj=coods[k]-coods[j]
                #        
                #        rkj_norm=np.linalg.norm(rkj)
#
                #        fcutik, fcutjk =  fcut(rik_norm, cutoff_r), fcut(rkj_norm, cutoff_r)      
                #        fcut_tot = fcutij*fcutik*fcutjk
                #        #fcut_tot=1.0
#
                #        threeb[0][j][k] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                #        threeb[1][j][k] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                #        threeb[2][j][k] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                #        
                #        atm = (rij_norm*rik_norm*rkj_norm)**2
                #        
                #        charge = np.cbrt(z*z2*z3)
                #        
                #        threeb[3][j][k], threeb[4][j][k], threeb[5][j][k] =  atm, charge, fcut_tot

    #threeb2 = np.zeros((size,size,size,6))
#
    #for i in range(len(threeb)):
    #    threeb2[:,:,:,i] = threeb[i] + threeb[i].T #above loops populated the upper tetrahedron of the rank 3 tensor, lower is populated by the transpose since the tensors are symmetric

    #return twob, threeb2
    return twob


def get_convolutions(rstep,rcut,alpha_list,n_list,order,a1,astep=None,nAs=None,normalized=False,gradients=False):
    """
    returns cMBDF convolutions evaluated via Fast Fourier Transforms
    """
    step_r = rcut/next_fast_len(int(rcut/rstep))
    rgrid = np.arange(0.0,rcut, step_r)
    size = len(rgrid)
    gaussian = np.exp(-a1*(rgrid**2))

    m = order+1

    temp1, temp2 = [], []
    dtemp1, dtemp2 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(rgrid,i,a1) for i in range(1,m+1)]]
    
    for i in range(m):
        fm = fms[i]
        #fm1 = fms[i+1]
        
        temp, dtemp = [], []
        for alpha in alpha_list:
            gn = np.exp(-alpha*rgrid)
            arr = fftconvolve(gn, fm, mode='full')[:size]*step_r
            temp.append(arr)
            darr = np.gradient(arr,step_r)
            #darr = -fftconvolve(gn, fm1, mode='full')[:size]*step_r
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

        temp, dtemp = [], []
        for n in n_list:
            gn = 2.2508*((rgrid+1)**n)
            arr = fftconvolve(1/gn, fm, mode='full')[:size]*step_r
            temp.append(arr)
            darr = np.gradient(arr,step_r)
            #darr = -fftconvolve(1/gn, fm1, mode='full')[:size]*step_r
            dtemp.append(darr)
        temp2.append(np.array(temp))
        dtemp2.append(np.array(dtemp))

    convs = np.concatenate((np.asarray(temp1),np.asarray(temp2)),axis=1)
    dconvs = np.concatenate((np.asarray(dtemp1),np.asarray(dtemp2)),axis=1)
    if normalized==False:
        return np.asarray([convs, dconvs])
    elif normalized=='same':
        norms = [np.max(np.abs(conv)) for conv in convs]
        return np.asarray([[convs[i]/norms[i] for i in range(len(convs))],
                        [dconvs[i]/norms[i] for i in range(len(convs))]])   
    elif normalized=='separate':
        norms1 = [np.max(np.abs(conv)) for conv in convs]
        norms2 = [np.max(np.abs(conv)) for conv in dconvs]
        return np.asarray([[convs[i]/norms1[i] for i in range(len(convs))],
                        [dconvs[i]/norms2[i] for i in range(len(convs))]])         


def get_cmbdf(charges, coods, rconvs, pad, rcut=12.0,n_atm = 2.0, aconvs=None, gradients=False):
    """"
    returns the local cMBDF representation for a molecule
    """
    size = len(charges)
    #nr, na = len(rconvs), len(aconvs)
    if gradients:
        m, n = rconvs.shape[1], rconvs.shape[2]
    else:
        m, n = rconvs.shape[0], rconvs.shape[1]
    nr = m*n
    #desc_size = nr+na
    desc_size = nr
    mat=np.zeros((pad,desc_size))

    assert size > 2, "No implementation for mono and diatomics yet"

    #twob, threeb = generate_data(size, charges,coods,rcut,n_atm)

    if gradients:
        dmat = np.zeros((pad,desc_size,pad,3))
        twob, twob_grad = generate_data_with_gradients(size, charges,coods,rconvs,rcut,n_atm)
        mat[:size] = einsum('ij... -> i...',twob)
        dmat[:size, :, :size, :] = twob_grad
        return mat, dmat
    
    else:
        twob = generate_data(size,charges,coods,rconvs,rcut,n_atm)
        mat[:size] = einsum('ij... -> i...',twob)
        return mat

from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,alpha_list,n_list,gradients=False,local=True,n_jobs=-1,a1=0.5,pad=None,rstep=0.01,rcut=12.0,astep=0.02,nAs=4,order=4,alpha=1.5,progress_bar=False,a2=None,normalized=True,n_atm = None):
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    #print(np.asarray(charges)[0].shape)
    #charges = np.asarray(charges)[0]
    #coords = coords[0]
    if pad==None:
        pad = max(lengths)

    rconvs = get_convolutions(rstep,rcut,alpha_list,n_list,order,a1,astep,nAs,normalized,gradients)
    #print(rconvs.shape)
    aconvs=None

    if local:
        if gradients:
            if progress_bar:
                from tqdm import tqdm
                mbdf = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, rconvs, pad, rcut,n_atm, aconvs, gradients=True) for charge,cood in tqdm(list(zip(charges,coords))))

            else:
                mbdf = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, rconvs, pad, rcut,n_atm, aconvs, gradients=True) for charge,cood in list(zip(charges,coords)))

            A, dA = [], []
            for i in range(len(mbdf)):
                A.append(mbdf[i][0])
                dA.append(mbdf[i][1])
            A, dA = np.array(A), np.array(dA)            

            return A, dA


        else:
            if progress_bar:
                from tqdm import tqdm
                reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, rconvs, pad, rcut,n_atm, aconvs) for charge,cood in tqdm(list(zip(charges,coords))))

            else:
                reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, rconvs, pad, rcut,n_atm, aconvs) for charge,cood in list(zip(charges,coords)))

            return np.asarray(reps)
