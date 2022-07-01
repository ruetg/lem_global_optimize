import multiprocess as mp
import matplotlib.pyplot as plt
import rasterio
import sys
#sys.path.insert(1, '/Users/gr_1/Documents/simplem/') #modify based on where it is
import simplem_par as simplem
import numpy as np
import ast

################ -------Params-------- ##################################
dem_folder = '/Users/gr_1/Google Drive_1/data_earth_engine/'
outfolder = 'results/v3_dA/'
# Number of processors
n_proc = 1 
# Number of simulations (parameter sets)
nr = 1000  
## Stream power parameters
#Stream power m/n (theta)
theta = 0.45  
# Vector of n values.  Currently n ranges from 0-4
ns = np.random.rand(nr) * 4.0 
# The ratio D/k. Currently the prior distribution is log-uniform from 0 to 10
diffus = 0 * np.power(10.0, np.random.rand(nr) * 12 + 1)  
#A_crit values range as a log-uniform distribution
careas = np.power(10.0, np.random.rand(nr) * 3) 
# diffusion exponents (p)
p = np.zeros(nr) + 1.0 
########################################################################


n_basin = 4631  # number of basins in octopus
ms = ns * theta # Vector of m values.  m depends on n

#The erosion rates for each basin, to contain a vector of avg erosion rates
eros1 = [None] * n_basin
#vector of avg slopes in each basin, to compare to octopus slopes
slps2 = np.zeros(n_basin)


def par_ero(i):
    """
    Parallel erosion routine for each DEM basin
    
    :param i: basin number
    :returns: A; accumulated erosion
    """
    
    #First run diffusion - we raise the coefficient by 1/p and then raise the whole
    #diffusion rate by p in order to
    E = simplem.diffuse(
        Zi, D=-(diffus[i]**(1.0 / p[i])), dy=dy1, dx=dx1, dt=1)
    E[E < 0] = 0
    #We only want the erosion part....
    E = E**p[i]
    #Raise too p
    m = ms[i]
    n = ns[i]
    #Now add the hillslope erosion and the fluvial erosion
    ero = simplem.erode_explicit(
        slps,
        I1,
        s1,
        A1,
        E,
        dx=dx1,
        dy=dy1,
        m=ms[i],
        n=ns[i],
        k=1e-8,
        carea=careas[i])
    
    #This should be zero anyways but just in case...
    ero[slps == 0] = 0
    #Sum erosion downstream = we do it this way so that nodes draining outside of the basin (i.e. on the edge) are not included
    A = simplem.acc(I1, s1, init=ero.copy())
    #Calculate the avg erosion per drainage area...
    pl = (A.ravel()[np.argmax(A1.ravel())]) / np.max(A1.ravel())
    return pl, i


for c in range(n_basin):
    try:
        dem = rasterio.open(dem_folder + 'hydrosheds_bas_v3_{}.tif'.format(str(c)))
        lat = dem.xy(0, 0)[1]
        dx = np.cos(lat / 180 * np.pi) * (1852 / 60) * \
            3  # dx is dependent on latitude
        f = simplem.simple_model()
        f.dx = dx
        f.dy = 92.59
        
        # We must pad the DEM in order to prevent edge effects
        f.set_z(np.pad(np.float64(np.squeeze(dem.read())), pad_width=2))
        
        #Outlet nodes are at or below 0
        f.BC = np.where(f.Z.transpose().ravel() <= 0)[0]
        
        #Fill local sinks
        f.sinkfill()
        
        #calculate local slopes and populate the receiver grid
        f.slp_basin()
        
        #Build the Fastscape stack
        f.stack()
        
        #calculate the receiver grid
        f.acc()
        
        #Get Elevation, corrected
        Zi = f.Z.copy()
        
        #Get drainage area
        A1 = f.A.copy()
        
        #Initialize mean erosion rate (per basin) vector
        mnmat = np.zeros((len(ms), 1))
        
        
        A1 = f.A.copy()
        ny1 = f.ny
        nx1 = f.nx
        I1 = f.I.copy()
        s1 = f.s.copy()
        dy1 = f.dy
        dx1 = dx
        with mp.Pool(n_proc) as p:
            vals = p.map(do_ero, [i for i in range(len(ns))])

        eros1[c] = mnmat[list(zip(*vals))[1], 0] = list(zip(*vals))[0]
        f.acc(slps)
        slpsall[c] = (f.A.ravel()[f.Z.ravel() > 0][np.argmax(
            A1.ravel()[f.Z.ravel() > 0])]) / np.max(A1.ravel()[f.Z.ravel() > 0])
    except Exception as e:
        print('err: ' + str(c))
        print(e)
if 1:
    np.save('{}/eros'.format(outfolder),eros1)
    np.save('{}/diffu'.format(outfolder),diffus)
    np.save('{}/ms'.format(outfolder),ms)
    np.save('{}/ns'.format(outfolder),ns)
    np.save('{}/careas'.format(outfolder),careas)
    np.save('{}/slps2'.format(folder),slpsall)
    np.save('{}/dns'.format(folder),p)