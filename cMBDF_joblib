import numpy as np
import numba as nb
from numpy import einsum
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len


@nb.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
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
def generate_data_with_gradients(size,charges,coods,rconvs_arr, aconvs_arr,cutoff_r=12.0,n_atm = 2.0):
    rconvs, drconvs = rconvs_arr[0], rconvs_arr[1]
    aconvs, daconvs = aconvs_arr[0], aconvs_arr[1]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1*n1
    nAs = m2*n2
    rstep = cutoff_r/rconvs.shape[-1]
    astep = np.pi/(aconvs.shape[-1])

    twob = np.zeros((size,size,nrs))
    threeb = np.zeros((size,size,size,nAs))
    twob_temps = np.zeros((size,size,6))
    threeb_temps = np.zeros((size,size,size,5))

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=np.linalg.norm(rij)

            if rij_norm!=0 and rij_norm<cutoff_r:
                grad_dist = rij/rij_norm
                z2 = charges[j]
                ind = rij_norm/rstep
                ac = np.sqrt(z*z2)
                pref = ac
                twob_temps[i][j][:3] = ac, rij_norm, ind
                twob_temps[j][i][:3] = ac, rij_norm, ind
                twob_temps[i][j][3:] = grad_dist
                twob_temps[j][i][3:] = -grad_dist             
                ind = int(ind)
                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref*rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2+=1

                for k in range(j+1, size):
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)

                        cos1 = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        cos2 = np.minimum(1.0,np.maximum(np.dot(-rij,-rkj)/(rij_norm*rkj_norm),-1.0))
                        cos3 = np.minimum(1.0,np.maximum(np.dot(rkj,-rik)/(rkj_norm*rik_norm),-1.0))

                        ang1 = np.arccos(cos1)
                        ang2 = np.arccos(cos2)
                        ang3 = np.arccos(cos3)
                        sin1, sin2, sin3 = np.abs(np.sin(ang1)), np.abs(np.sin(ang2)), np.abs(np.sin(ang3))
                        
                        ind1 = ang1/astep
                        ind2 = ang2/astep
                        ind3 = ang3/astep

                        atm = np.exp(n_atm*(rij_norm + rik_norm + rkj_norm))

                        charge = np.cbrt(z*z2*z3)
                        
                        pref = charge

                        threeb_temps[i][j][k] = charge, atm, ind1, sin1, cos1
                        threeb_temps[i][k][j] = charge, atm, ind1, sin1, cos1
                        threeb_temps[j][i][k] = charge, atm, ind2, sin2, cos2
                        threeb_temps[j][k][i] = charge, atm, ind2, sin2, cos2
                        threeb_temps[k][i][j] = charge, atm, ind3, sin3, cos3
                        threeb_temps[k][j][i] = charge, atm, ind3, sin3, cos3

                        ind1, ind2, ind3 = int(ind1), int(ind2), int(ind3)

                        id2=0
                        for i1 in range(m2):
                            for i2 in range(n2):
                                conv1 = (pref*aconvs[i1][i2][ind1])/atm
                                conv2 = (pref*aconvs[i1][i2][ind2])/atm
                                conv3 = (pref*aconvs[i1][i2][ind3])/atm
                                
                                threeb[i][j][k][id2] = conv1
                                threeb[i][k][j][id2] = conv1

                                threeb[j][i][k][id2] = conv2
                                threeb[j][k][i][id2] = conv2

                                threeb[k][j][i][id2] = conv3
                                threeb[k][i][j][id2] = conv3

                                id2+=1

    twob_grad = np.zeros((size,nrs,size,3))
    threeb_grad = np.zeros((size,nAs,size,3))

    for i in range(size):
        grad_temp = np.zeros((nrs,3))
        agrad_temp = np.zeros((nAs,3))

        for j in range(size):
            rij = coods[i] - coods[j]
            agrad_temp2 = np.zeros((nAs,3))

            if j!=i:
                
                ac, rij_norm, ind = twob_temps[i][j][:3]
                grad_dist = -twob_temps[i][j][3:]
                if ac!=0:
                    ind = int(ind)

                    pref = ac
                    id2 = 0

                    for i1 in range(m1):
                        for i2 in range(n1):
                            grad1 = pref*drconvs[i1][i2][ind]*grad_dist
                            twob_grad[i][id2][j] = grad1

                            grad_temp[id2] += -grad1
                            id2+=1

                for k in range(size):
                    if k!=j and k!=i:
                        rkj = coods[k] - coods[j]
                        rik = coods[i] - coods[k]
                        rik_norm = twob_temps[i][k][1]

                        ac, atm, ind1, sin1, cos1  = threeb_temps[i][j][k]
                        if ac!=0:
                            grad_distjk, grad_distik = twob_temps[j][k][3:], twob_temps[i][k][3:]
                            ind1  = int(ind1)

                            atm_gradi = -n_atm*(grad_distik - grad_dist)/atm
                            atm_gradj = -n_atm*(grad_distjk + grad_dist)/atm

                            if sin1==0:
                                gang1i, gang1j = np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0])
                            else:
                                gang1i = -((cos1*(-(rij_norm*grad_distik) + (rik_norm*grad_dist))) + ((rij+ rik)))/(sin1*rij_norm*rik_norm)
                                         #-((cos1*(-(rij_norm*grad_distik) - (rik_norm*grad_dist))) + ((rij+ rik)))/(sin1*rij_norm*rik_norm)
                                gang1j = -(-rik - (grad_dist*cos1*rik_norm))/(sin1*rij_norm*rik_norm)
                                         #-(-rik + (grad_dist*cos1*rik_norm))/(sin1*rij_norm*rik_norm)
                            id2=0
                            for i1 in range(m2):
                                for i2 in range(n2):
                                    af, daf = aconvs[i1][i2][ind1], daconvs[i1][i2][ind1] 
                                    grad1 = ac*daf
                                    grad3 = ac*af
                                    agrad_temp2[id2] += (grad1*gang1j)/atm + (grad3*atm_gradj)
                                    agrad_temp[id2] += (grad1*gang1i)/atm + (grad3*atm_gradi)
                                    id2+=1
                
            threeb_grad[i,:,j,:] = agrad_temp2

        twob_grad[i,:,i,:] = grad_temp

        threeb_grad[i,:,i,:] = agrad_temp
   
    return twob, twob_grad, threeb, threeb_grad


@nb.jit(nopython=True)
def generate_data(size,charges,coods,rconvs,aconvs,cutoff_r=12.0,n_atm = 2.0):
    rconvs, aconvs = rconvs[0], aconvs[0]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1*n1
    nAs = m2*n2
    rstep = cutoff_r/rconvs.shape[-1]
    astep = np.pi/aconvs.shape[-1]
    twob = np.zeros((size,size,nrs))
    threeb = np.zeros((size,size,size,nAs))

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=np.linalg.norm(rij)

            if rij_norm!=0 and rij_norm<cutoff_r:
                z2 = charges[j]
                ind = int(rij_norm/rstep)
                pref = np.sqrt(z*z2)
                
                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref*rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2+=1

                for k in range(j+1, size):
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)

                        cos1 = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        cos2 = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        cos3 = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        ang1 = np.arccos(cos1)
                        ang2 = np.arccos(cos2)
                        ang3 = np.arccos(cos3)
                        
                        ind1 = int(ang1/astep)
                        ind2 = int(ang2/astep)
                        ind3 = int(ang3/astep)

                        atm = (rij_norm*rik_norm*rkj_norm)**n_atm
                        
                        charge = np.cbrt(z*z2*z3)
                        
                        pref = charge

                        id2=0
                        for i1 in range(m2):
                            for i2 in range(n2):
                                if i2==1:
                                    conv1 = (pref*aconvs[i1][i2][ind1]*cos2*cos3)/atm
                                    conv2 = (pref*aconvs[i1][i2][ind2]*cos1*cos3)/atm
                                    conv3 = (pref*aconvs[i1][i2][ind3]*cos2*cos1)/atm
                                else:
                                    conv1 = (pref*aconvs[i1][i2][ind1])/atm
                                    conv2 = (pref*aconvs[i1][i2][ind2])/atm
                                    conv3 = (pref*aconvs[i1][i2][ind3])/atm
                                
                                threeb[i][j][k][id2] = conv1
                                threeb[i][k][j][id2] = conv1

                                threeb[j][i][k][id2] = conv2
                                threeb[j][k][i][id2] = conv2

                                threeb[k][j][i][id2] = conv3
                                threeb[k][i][j][id2] = conv3

                                id2+=1

    return twob,threeb 
         

def get_cmbdf(charges, coods, rconvs, pad, rcut=12.0,n_atm = 2.0, aconvs=None, gradients=False):
    """"
    returns the local cMBDF representation for a molecule
    """
    size = len(charges)
    #nr, na = len(rconvs), len(aconvs)
    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]
    nr = m1*n1
    na = m2*n2
    desc_size = nr+na
    mat=np.zeros((pad,desc_size))

    assert size > 2, "No implementation for mono and diatomics yet"

    if gradients:
        dmat = np.zeros((pad,desc_size,pad,3))
        twob, twob_grad, threeb, threeb_grad = generate_data_with_gradients(size, charges,coods,rconvs, aconvs,rcut,n_atm)
        mat[:size,:nr] = einsum('ij... -> i...',twob)
        mat[:size,nr:] = einsum('ijk... -> i...',threeb)
        dmat[:size, :nr, :size, :] = twob_grad
        dmat[:size, nr:, :size, :] = threeb_grad
        return mat, dmat
    
    else:
        twob, threeb = generate_data(size,charges,coods,rconvs,aconvs,rcut,n_atm)
        mat[:size,:nr] = einsum('ij... -> i...',twob)
        mat[:size,nr:] = einsum('ijk... -> i...',threeb)
        return mat


def get_convolutions(rstep,rcut,alpha_list,n_list,order,a1,a2,astep=None,nAs=None,normalized=False,gradients=False):
    """
    returns cMBDF convolutions evaluated via Fast Fourier Transforms
    """
    step_r = rcut/next_fast_len(int(rcut/rstep))
    #astep = np.pi/next_fast_len(int(np.pi/astep))
    astep = 0.0002
    rgrid = np.arange(0.0,rcut, step_r)
    agrid = np.arange(0.0,np.pi,astep)

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

    rconvs = np.concatenate((np.asarray(temp1),np.asarray(temp2)),axis=1)
    drconvs = np.concatenate((np.asarray(dtemp1),np.asarray(dtemp2)),axis=1)

    size = len(agrid)
    gaussian = np.exp(-a1*(agrid**2))

    m = order+1

    temp1, dtemp1 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(agrid,i,a2) for i in range(1,m+1)]]
    
    for i in range(m):
        fm = fms[i]
        
        temp, dtemp = [], []
        for n in range(1,nAs+1):
            gn = np.cos(n*agrid)
            arr = fftconvolve(gn, fm, mode='full')[:size]*astep
            temp.append(arr)
            darr = np.gradient(arr,astep)
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

    aconvs, daconvs = np.asarray(temp1), np.asarray(dtemp1)

    if normalized==False:
        return np.asarray([rconvs, drconvs]), np.asarray([aconvs, daconvs])
    
    else:
        rnorms = [np.max(np.abs(conv)) for conv in rconvs]
        anorms = [np.max(np.abs(conv)) for conv in aconvs]
        return np.asarray([[rconvs[i]/rnorms[i] for i in range(len(rconvs))],
                        [drconvs[i]/rnorms[i] for i in range(len(rconvs))]]), np.asarray([[aconvs[i]/anorms[i] for i in range(len(aconvs))],
                        [daconvs[i]/anorms[i] for i in range(len(aconvs))]]) 
    

from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,alpha_list=[1.5,5.0],n_list=[3.0,5.0],gradients=False,local=True,n_jobs=-1,a1=2.0,pad=None,rstep=0.001,rcut=8.0,astep=0.0002,nAs=4,order=4,alpha=1.5,progress_bar=False,a2=2.0,normalized='same',n_atm = 1.0):
                                        
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    rconvs, aconvs = get_convolutions(rstep,rcut,alpha_list,n_list,order,a1,a2,astep,nAs,normalized,gradients)

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
