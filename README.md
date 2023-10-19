# cMBDF
comvolutional MBDF via fast fourier transforms
Usage :

'''
from cMBDF import get_convolutions, get_cmbdf

convs = cMBDF_fft.get_convolutions() #only needs to be done once, can also store it as .npy 

pad_size = max([len(mol.charges) for mol in mols])
rep, drep = [], []
for mol in mols:
    r, dr = cMBDF_fft.get_cmbdf(mol.charges,mol.coordinates,convs,pad_size,gradients=True)
    rep.append(r)
    drep.append(dr)
'''
