import numpy as np
import numba as nb
from numpy import einsum
from numpy.linalg import norm as norm_
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len

rvdw = np.array((1.0,  # Ghost atom
    1.20,       #  1 H
    1.40,       #  2 He [1]
    1.82,       #  3 Li [1]
    1.53,       #  4 Be [5]
    1.92,       #  5 B  [5]
    1.70,       #  6 C  [1]
    1.55,       #  7 N  [1]
    1.52,       #  8 O  [1]
    1.47,       #  9 F  [1]
    1.54,       # 10 Ne [1]
    2.27,       # 11 Na [1]
    1.73,       # 12 Mg [1]
    1.84,       # 13 Al [5]
    2.10,       # 14 Si [1]
    1.80,       # 15 P  [1]
    1.80,       # 16 S  [1]
    1.75,       # 17 Cl [1]
    1.88,       # 18 Ar [1]
    2.75,       # 19 K  [1]
    2.31,       # 20 Ca [5]
    1.0,    # 21 Sc
    1.0,    # 22 Ti
    1.0,    # 23 V
    1.0,    # 24 Cr
    1.0,    # 25 Mn
    1.0,    # 26 Fe
    1.0,    # 27 Co
    1.63,       # 28 Ni [1]
    1.40,       # 29 Cu [1]
    1.39,       # 30 Zn [1]
    1.87,       # 31 Ga [1]
    2.11,       # 32 Ge [5]
    1.85,       # 33 As [1]
    1.90,       # 34 Se [1]
    1.85,       # 35 Br [1]
    2.02,       # 36 Kr [1]
    3.03,       # 37 Rb [5]
    2.49,       # 38 Sr [5]
    1.0,    # 39 Y
    1.0,    # 40 Zr
    1.0,    # 41 Nb
    1.0,    # 42 Mo
    1.0,    # 43 Tc
    1.0,    # 44 Ru
    1.0,    # 45 Rh
    1.63,       # 46 Pd [1]
    1.72,       # 47 Ag [1]
    1.58,       # 48 Cd [1]
    1.93,       # 49 In [1]
    2.17,       # 50 Sn [1]
    2.06,       # 51 Sb [5]
    2.06,       # 52 Te [1]
    1.98,       # 53 I  [1]
    2.16,       # 54 Xe [1]
    3.43,       # 55 Cs [5]
    2.49,       # 56 Ba [5]
    1.0,    # 57 La
    1.0,    # 58 Ce
    1.0,    # 59 Pr
    1.0,    # 60 Nd
    1.0,    # 61 Pm
    1.0,    # 62 Sm
    1.0,    # 63 Eu
    1.0,    # 64 Gd
    1.0,    # 65 Tb
    1.0,    # 66 Dy
    1.0,    # 67 Ho
    1.0,    # 68 Er
    1.0,    # 69 Tm
    1.0,    # 70 Yb
    1.0,    # 71 Lu
    1.0,    # 72 Hf
    1.0,    # 73 Ta
    1.0,    # 74 W
    1.0,    # 75 Re
    1.0,    # 76 Os
    1.0,    # 77 Ir
    1.75,       # 78 Pt [1]
    1.66,       # 79 Au [1]
    1.55,       # 80 Hg [1]
    1.96,       # 81 Tl [1]
    2.02,       # 82 Pb [1]
    2.07,       # 83 Bi [5]
    1.97,       # 84 Po [5]
    2.02,       # 85 At [5]
    2.20,       # 86 Rn [5]
    3.48,       # 87 Fr [5]
    2.83,       # 88 Ra [5]
    1.0,    # 89 Ac
    1.0,    # 90 Th
    1.0,    # 91 Pa
    1.86,       # 92 U [1]
    1.0,    # 93 Np
    1.0,    # 94 Pu
    1.0,    # 95 Am
    1.0,    # 96 Cm
    1.0,    # 97 Bk
    1.0,    # 98 Cf
    1.0,    # 99 Es
    1.0,    #100 Fm
    1.0,    #101 Md
    1.0,    #102 No
    1.0,    #103 Lr
))


@nb.jit(nopython=True)
def basis(r,elem,a=1.0):
    a1 = (rvdw[int(elem)]*a)**2
    norm = np.sqrt((2*np.pi)*a1)
    gaussian = np.exp(-a1*(r**2))/norm
    return gaussian


@nb.jit(nopython=True)
def gaussian_product(r1, a1, r2, a2):
    mu = (r1 * a2 + r2 * a1) / (a1 + a2)
    sigma = (a1 * a2) / (a1 + a2)
    zeta = np.exp(- (r1 - r2)**2 / (a1 + a2))
    return mu, sigma, zeta


@nb.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 + 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2
    elif degree == 5:
        x1 = (a*x)**5
        x2 = (a*x)**3
        return -32*x1 + 160*x2 - 120*(a*x)
    
    
@nb.jit(nopython=True)
def chebyshev_polynomial(x, degree):
    if degree==1:
        return x
    elif degree==2:
        return 2*(x**2) - 1
    elif degree==3:
        return 4*(x**3) - 3*x
    elif degree==4:
        return 8*((x**4) - (x**2)) + 1
    elif degree==5:
        return 16*(x**5) - 20*(x**3) + 5*x


@nb.jit(nopython=True)
def fcut(Rij, rcut): #checked
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)


@nb.jit(nopython=True)
def fcut_with_grad(Rij, rcut):
    arg = (np.pi*Rij)/rcut
    return 0.5*(np.cos(arg)+1), (-np.pi*np.sin(arg))/(2*rcut)


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
    twob_temps = np.zeros((size,size,8))
    threeb_temps = np.zeros((size,size,size,9))

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=norm_(rij)

            if rij_norm!=0 and rij_norm<cutoff_r:
                grad_dist = rij/rij_norm
                z2 = charges[j]
                arg = (np.pi*rij_norm)/cutoff_r
                fcutij, gfcut = 0.5*(np.cos(arg)+1), (-np.pi*np.sin(arg))/(2*cutoff_r)
                ind = rij_norm/rstep
                ac = np.sqrt(z*z2)
                pref = ac*fcutij
                twob_temps[i][j][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[j][i][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[i][j][5:] = grad_dist
                twob_temps[j][i][5:] = -grad_dist             
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
                    rik_norm=norm_(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=norm_(rkj)

                        fcutik, fcutjk =  fcut(rik_norm, cutoff_r), fcut(rkj_norm, cutoff_r)      
                        fcut_tot = fcutij*fcutik*fcutjk

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

                        atm_temp = rij_norm*rik_norm*rkj_norm
                        atm = atm_temp**n_atm
                        atm1 = atm*atm_temp
                        atm = np.exp(n_atm*(rij_norm + rik_norm + rkj_norm))

                        charge = np.cbrt(z*z2*z3)
                        
                        pref = charge*fcut_tot

                        #atm=1.0
                        threeb_temps[i][j][k] = charge, pref, atm, atm1, ind1, sin1, cos1, cos2, cos3
                        threeb_temps[i][k][j] = charge, pref, atm, atm1, ind1, sin1, cos1, cos3, cos2
                        threeb_temps[j][i][k] = charge, pref, atm, atm1, ind2, sin2, cos2, cos1, cos3
                        threeb_temps[j][k][i] = charge, pref, atm, atm1, ind2, sin2, cos2, cos3, cos1
                        threeb_temps[k][i][j] = charge, pref, atm, atm1, ind3, sin3, cos3, cos1, cos2
                        threeb_temps[k][j][i] = charge, pref, atm, atm1, ind3, sin3, cos3, cos2, cos1

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
                
                ac, fcutij, gfcutij, rij_norm, ind = twob_temps[i][j][:5]
                grad_dist = -twob_temps[i][j][5:]
                if ac!=0:
                    ind = int(ind)

                    gradfcut = gfcutij*grad_dist

                    pref = ac*fcutij
                    id2 = 0

                    for i1 in range(m1):
                        for i2 in range(n1):
                            grad1 = pref*drconvs[i1][i2][ind]*grad_dist
                            grad2 = (twob[i][j][id2]*gradfcut)/fcutij
                            twob_grad[i][id2][j] = grad1+grad2

                            grad_temp[id2] += -(grad1+grad2)
                            id2+=1

                for k in range(size):
                    if k!=j and k!=i:
                        rkj = coods[k] - coods[j]
                        rik = coods[i] - coods[k]
                        fcutik, gfcutik, rik_norm = twob_temps[i][k][1:4]
                        fcutjk, gfcutjk, rjk_norm = twob_temps[j][k][1:4]
                        ac, pref, atm, atm1, ind1, sin1, cos1, cos2, cos3  = threeb_temps[i][j][k]
                        if ac!=0:
                            grad_distjk, grad_distik = twob_temps[j][k][5:], twob_temps[i][k][5:]
                            ind1  = int(ind1)

                            temp = gradfcut*fcutik*fcutjk
                            gradfcuti = -temp + (gfcutik*grad_distik*fcutij*fcutjk)
                            gradfcutj = temp + (gfcutjk*grad_distjk*fcutik*fcutij)

                            atm_gradi = -n_atm*(grad_distik - grad_dist)/atm
                            atm_gradj = -n_atm*(grad_distjk + grad_dist)/atm
                            
                            denom = sin1*rij_norm*rik_norm
                            if denom==0:
                                denom+=1e-8 #dirty fix makes me wanna throw upppppp
                            gang1i = -((cos1*(-(rij_norm*grad_distik) + (rik_norm*grad_dist))) + ((rij+ rik)))/denom
                            gang1j = -(-rik - (grad_dist*cos1*rik_norm))/denom

                            id2=0
                            for i1 in range(m2):
                                for i2 in range(n2):
                                    af, daf = aconvs[i1][i2][ind1], daconvs[i1][i2][ind1] 
                                    grad1 = pref*daf
                                    grad2 = ac*af
                                    grad3 = pref*af
                                    agrad_temp2[id2] += ((grad1*gang1j)+(grad2*gradfcutj))/atm + (grad3*atm_gradj)
                                    agrad_temp[id2] += ((grad1*gang1i)+(grad2*gradfcuti))/atm + (grad3*atm_gradi)
                                    id2+=1

            threeb_grad[i,:,j,:] = agrad_temp2

        twob_grad[i,:,i,:] = grad_temp

        threeb_grad[i,:,i,:] = agrad_temp
   
    return twob, twob_grad, threeb, threeb_grad


@nb.jit(nopython=True)
def generate_data(size,charges,coods,elems,rconvs,aconvs,tconvs,cutoff_r=12.0,n_atm = 2.0,natm2 = 1.0,a3 = 1.0):
    aconvs = aconvs[0]
    tconvs = tconvs[0]
    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1*n1
    nAs = m2*n2
    m3, n3 = tconvs.shape[0], tconvs.shape[1]
    nds = m3*n3
    rstep = cutoff_r/rconvs.shape[-1]
    astep = np.pi/aconvs.shape[-1]
    dmat = np.zeros((size, size))
    twob = np.zeros((size,size,nrs))
    threeb = np.zeros((size,size,size,nAs))
    fourb = np.zeros((size, size, size, size, nds))

    for i in range(size):
        z, atom = charges[i], coods[i]

        for j in range(i+1,size):
            rij=atom-coods[j]
            rij_norm=norm_(rij)

            dmat[i,j] = rij_norm
            dmat[j,i] = rij_norm

            if rij_norm!=0 and rij_norm<cutoff_r:
                z2 = charges[j]
                #print(z2)
                #ind = int(rij_norm/rstep)
                ind = int(np.rint(rij_norm/rstep))
                pref = np.sqrt(z*z2)
                
                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref*rconvs[np.where(elems==z2)[0][0]][i1][i2][ind]
                        #print(conv)
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2+=1

                for k in range(j+1, size):
                    rik=atom-coods[k]
                    rik_norm=norm_(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=norm_(rkj)

                        cos1 = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        cos2 = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        cos3 = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        ang1 = np.arccos(cos1)
                        ang2 = np.arccos(cos2)
                        ang3 = np.arccos(cos3)
                        
                        #ind1 = int(ang1/astep)
                        ind1 = int(np.rint(ang1/astep))
                        #ind2 = int(ang2/astep)
                        ind2 = int(np.rint(ang2/astep))
                        #ind3 = int(ang3/astep)
                        ind3 = int(np.rint(ang3/astep))
                        
                        charge = np.cbrt(z*z2*z3)
                        
                        pref = charge
                        #pref = charge
                        atm = (rij_norm*rik_norm*rkj_norm)**n_atm

                        id2=0
                        for i1 in range(m2):
                            for i2 in range(n2):
                                if i2==0:
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

    for i in range(size):
    
        z, atom = charges[i], coods[i]

        for j in range(size):

            rij_norm = dmat[i,j]

            z2 = charges[j]

            for k in range(size):
                
                rik_norm = dmat[i,k]
                z3 = charges[k]
                
                for l in range(size):
                    ril_norm = dmat[i,l]
                    z4 = charges[l]

                    if np.sum(np.array([0<rij_norm<cutoff_r, 0<rik_norm<cutoff_r, 0<ril_norm<cutoff_r])) == 3:
                        
                        charge = np.sqrt(np.sqrt(z*z2*z3*z4))
                        r1, r2, r3 = rij_norm, rik_norm, ril_norm
                        r4, r5, r6 = dmat[j,k], dmat[j,l], dmat[k,l]
                        
                        mu1, sigma1, zeta1 = gaussian_product(r1,a3,r2,a3)

                        for r in [r3,r4,r5,r6]:
                            mu1, sigma1, zeta2 = gaussian_product(mu1,sigma1,r,a3)
                            zeta1 = zeta1*zeta2

                
                        pref = charge*zeta1/((r1*r2*r3)**natm2)
                        ind = int(np.rint(mu1/rstep))

                        id = 0
                        for i1 in range(m3):
                            for i2 in range(n3):
                                conv = pref*tconvs[i1][i2][ind]
                                fourb[i,j,k,l][id] = conv
                                id+=1

    return twob,threeb,fourb 


def get_cmbdf(charges, coods, convs, elems, pad=None, rcut=10.0,n_atm = 1.0,natm2 = 1.0, gradients=False,a3 = 1.0):
    """"
    returns the local cMBDF representation for a molecule
    """
    rconvs, aconvs, tconvs = convs
    size = len(charges)
    if pad==None:
        pad = size
    #nr, na = len(rconvs), len(aconvs)
    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    
    m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]

    m3, n3 = tconvs[0].shape[0], tconvs[0].shape[1]

    nr = m1*n1
    na = m2*n2
    nd = m3*n3
    desc_size = nr+na+nd
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
        twob, threeb, fourb = generate_data(size,charges,coods,elems,rconvs,aconvs,tconvs,rcut,n_atm,natm2=natm2,a3=a3)
        mat[:size,:nr] = einsum('ijk -> ik',twob)
        mat[:size,nr:nr+na] = einsum('ijkl -> il',threeb)
        mat[:size,nr+na:] = einsum('ijklm -> im',fourb)
        return mat
    
def get_cmbdf_global(charges, coods, asize,rep_size,keys, convs, elems,rcut=10.0,n_atm = 1.0,natm2 = 1.0, gradients=False,a3 = 1.0):
    """
    returns the flattened, bagged cMBDF feature vector for a molecule
    """
    rconvs, aconvs, tconvs = convs
    #print(rconvs.shape, aconvs.shape)
    size = len(charges)

    assert size > 2, "No implementation for mono and diatomics yet"

    elements = {k:[] for k in keys}

    for i in range(size):
        elements[charges[i]].append(i)

    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    
    m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]

    m3, n3 = tconvs[0].shape[0], tconvs[0].shape[1]

    nr = m1*n1
    na = m2*n2
    nd = m3*n3
    desc_size = nr+na+nd

    mat, ind = np.zeros((rep_size,desc_size)), 0

    twob, threeb, fourb = generate_data(size,charges,coods,elems,rconvs,aconvs,tconvs,rcut,n_atm,natm2=natm2,a3=a3)

    for key in keys:

        num = len(elements[key])

        bags = np.zeros((num,desc_size))

        if num!=0:

            bags[:,:nr] = einsum('ijk -> ik',twob[elements[key]])
            bags[:,nr:nr+na] = einsum('ijkl -> il',threeb[elements[key]])
            bags[:,nr+na:] = einsum('ijklm -> im',fourb[elements[key]])

            mat[ind:ind+num] = -np.sort(-bags,axis=0)

        ind += asize[key]

    return mat.ravel(order='F')

def get_convolutions(elems,rstep=0.0008,rcut=10.0,alpha_list=[1.5,5.0],n_list=[3.0,5.0],order=4,a1=2.0,a2=2.0,a3 = 1.0,alpha2 = 1.5,astep=0.0002,nAs=4,gradients=True):
    """
    returns cMBDF convolutions evaluated via Fast Fourier Transforms
    """
    a3 = 1.0
    step_r = rcut/next_fast_len(int(rcut/rstep))
    astep = np.pi/next_fast_len(int(np.pi/astep))
    rgrid = np.arange(0.0,rcut, step_r)
    rgrid2 = np.arange(-rcut,rcut, step_r)
    agrid = np.arange(0.0,np.pi,astep)
    agrid2 = np.arange(-np.pi,np.pi,astep)

    size = len(rgrid)

    rconvs = []

    for elem in elems:
        a1 = rvdw[int(elem)]**2
        norm = np.sqrt((2*np.pi)*a1)
        gaussian = np.exp(-a1*(rgrid2**2))/norm
        m = order+1

        temp1, temp2 = [], []
        dtemp1, dtemp2 = [], []

        fms = [gaussian, *[gaussian*hermite_polynomial(rgrid2,i,a1) for i in range(1,m+1)]]

        for i in range(m):
            fm = fms[i]

            temp, dtemp = [], []
            for alpha in alpha_list:
                gn = np.exp(-alpha*rgrid)
                arr = fftconvolve(gn, fm, mode='same')*step_r
                arr = arr/np.max(np.abs(arr))
                temp.append(arr)
                darr = np.gradient(arr,step_r)
                dtemp.append(darr)
            temp1.append(np.array(temp))
            dtemp1.append(np.array(dtemp))

            temp, dtemp = [], []
            for n in n_list:
                gn = 2.2508*((rgrid+1)**n)
                arr = fftconvolve(1/gn, fm, mode='same')[:size]*step_r
                arr = arr/np.max(np.abs(arr))
                temp.append(arr)
                darr = np.gradient(arr,step_r)
                dtemp.append(darr)
            temp2.append(np.array(temp))
            dtemp2.append(np.array(dtemp))
        rconvs.append(np.concatenate((np.asarray(temp1),np.asarray(temp2)),axis=1))
    drconvs = np.concatenate((np.asarray(dtemp1),np.asarray(dtemp2)),axis=1)

    size = len(agrid)
    gaussian = np.exp(-a2*(agrid2**2))

    m = order+1

    temp1, dtemp1 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(agrid2,i,a2) for i in range(1,m+1)]]
    
    for i in range(m):
        fm = fms[i]
        
        temp, dtemp = [], []
        for n in range(1,nAs+1):
            gn = np.cos(n*agrid)
            #gn = chebyshev_polynomial(agrid,n)
            arr = fftconvolve(gn, fm, mode='same')*astep
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr,astep)
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

    aconvs, daconvs = np.asarray(temp1), np.asarray(dtemp1)

    gaussian = np.exp(-(rgrid2**2)/(a3/6))

    m = order+1

    temp1, dtemp1 = [], []

    fms = [gaussian, *[gaussian*hermite_polynomial(rgrid2,i,1/(a3/6)) for i in range(1,m+1)]]
    
    for i in range(m):
        fm = fms[i]
        
        temp, dtemp = [], []
        for n in range(3):
            gn = np.exp(-alpha2*(n+1)*rgrid)
            arr = fftconvolve(gn, fm, mode='same')*step_r
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr,astep)
            dtemp.append(darr)

        for n in range(2):
            gn = 2.2508*((rgrid+1)**(n+2))
            arr = fftconvolve(1/gn, fm, mode='same')[:size]*step_r
            arr = arr/np.max(np.abs(arr))
            temp.append(arr)
            darr = np.gradient(arr,astep)
            dtemp.append(darr)
        temp1.append(np.array(temp))
        dtemp1.append(np.array(dtemp))

    tconvs, dtconvs = np.asarray(temp1), np.asarray(dtemp1)

    return np.asarray(rconvs), np.asarray([aconvs, daconvs]), np.asarray([tconvs, dtconvs])

from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,convs='None',alpha_list=[1.5,5.0],n_list=[3.0,5.0],alpha2 = 1.5,gradients=False,local=True,n_jobs=-1,a1=2.0,pad=None,rstep=0.0008,rcut=10.0,astep=0.0002,nAs=4,order=4,progress_bar=False,a2=2.0,n_atm = 2.0,natm2=1.0,a3 =1.0):
      
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    elems = np.unique(np.concatenate(charges))
    #(elems)
    if pad==None:
        pad = max(lengths)
    if type(convs)==str:
        convs = get_convolutions(elems,rstep,rcut,alpha_list,n_list,order,a1,a2,a3,alpha2,astep,nAs,gradients)

    if local:
        if gradients:
            if progress_bar:
                from tqdm import tqdm
                mbdf = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, convs, pad, rcut,n_atm, gradients=True, natm2 = natm2) for charge,cood in tqdm(list(zip(charges,coords))))

            else:
                mbdf = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, convs, pad, rcut,n_atm, gradients=True, natm2 = natm2) for charge,cood in list(zip(charges,coords)))

            A, dA = [], []
            for i in range(len(mbdf)):
                A.append(mbdf[i][0])
                dA.append(mbdf[i][1])
            A, dA = np.array(A), np.array(dA)            

            return A, dA


        else:
            if progress_bar:
                from tqdm import tqdm
                reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, convs, elems, pad, rcut,n_atm, gradients=False,natm2 = natm2, a3=a3) for charge,cood in tqdm(list(zip(charges,coords))))

            else:
                reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf)(charge, cood, convs, elems, pad, rcut,n_atm, gradients=False,natm2 = natm2, a3 = a3) for charge,cood in list(zip(charges,coords)))

            return np.asarray(reps)
        
    else:
        keys = np.unique(np.concatenate(charges))

        asize = {key:max([(mol == key).sum() for mol in charges]) for key in keys}

        rep_size = sum(asize.values())

        if progress_bar:
            from tqdm import tqdm
            reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf_global)(charge, cood,asize,rep_size,keys, convs, elems,rcut,n_atm,gradients=False,natm2 = natm2, a3=a3) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            reps = Parallel(n_jobs=n_jobs)(delayed(get_cmbdf_global)(charge, cood, asize,rep_size,keys, convs,elems,rcut,n_atm,gradients=False,natm2 = natm2, a3=a3) for charge,cood in list(zip(charges,coords)))
        
        return np.asarray(reps)  

