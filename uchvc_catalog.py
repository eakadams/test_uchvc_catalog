#!/usr/bin/python

#Here are functions for calling w/in notebooks to produce things for paper

import sys
sys.path.append('/Users/adams/python')
import alfalfa
import numpy as np
import scipy.optimize as optimization
import aplpy
from matplotlib import rc
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
from astropy.io import ascii
import colormaps as cmaps
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, Column
from astropy.io import ascii

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

lg_colnames = ['Name', 'RAh','RAm','RAs','Decd','Decm','Decs','E(B-V)','(m-M)o','(m-M)o_err+','(m-M)o_err-',
           'vh','vh_err+','vh_err-','Vmag','Vmag_err+','Vmag_err-','PA','PA_err+','PA_err-',
           'e','e_err+','e_err-','muVo','muVo_err+','muVo_err-','rh','rh_err+','rh_err-',
           'sigma_s','sigma_s_err+','sigma_s_err-','vrot_s','vrot_s_err+','vrot_s_err-',
           'M_HI','sigma_g','sigma_g_err+','sigma_g_err-','vrot_g','vrot_g_err+','vrot_g_err-',
           '[Fe/H]','[Fe/H]_err+','[Fe/H]_err-','F','References']

lg_colstarts =(0,18,21,24,29,33,36,39,45,51,56,62,68,73,78,83,87,91,97,102,107,112,117,122,127,131,135,142,
               148,154,159,164,169,174,179,184,189,194,199,204,210,215,220,226,231,236,238)

#lg_colends = (17,19,22,27,31,34,37,43,49,54,59,66,71,76,81,85,89,95,100,105,110,115,120,125,129,133,140,146,
#              152,157,162,167,172,177,182,187,192,197,202,208,213,218,224,229,234,236,279)

def get_wvw(name,complex=False):
    #read file in
    tmp=ascii.read(name)
    wvw_gl=tmp['L']
    wvw_gb=tmp['B']
    c=coord.Galactic(l=wvw_gl,b=wvw_gb,unit=(u.degree,u.degree))
    crd = c.transform_to(coord.ICRS)
    wvw_ra=crd.ra.degree
    wvw_dec=crd.dec.degree
    wvw_cz,wvw_vlsr,wvw_vgsr,wvw_vlg,wvw_vdev=alfalfa.get_vels(name) 
    #note that I don't know if this calculates vels or takes this from file
    #but everything except cz is in file
    #So I should do a comparison at some point in time (but not now)
    if complex == False:
        wvw_cat = Table([tmp['Count'],wvw_ra,wvw_dec,wvw_gl,wvw_gb,wvw_cz,wvw_vlsr,wvw_vgsr,wvw_vlg,wvw_vdev,tmp['Flux']],
                names=('ID','RA','Dec','l','b','cz','vlsr','vgsr','vlg','vdev','flux'))
    else:
        ind1 = np.where(tmp['cpx'] != 'CR')
        ind2 = np.where(tmp['cpx'] != 'EN')
        ind3 = np.where(tmp['cpx'] != 'EP')
        ind4 = np.where(tmp['cpx'] != 'P')
        ind5 = np.intersect1d(ind1,ind2)
        ind6=np.intersect1d(ind5,ind3)
        ind_cpx=np.intersect1d(ind6,ind4)
        wvw_cat = Table([tmp['Count'][ind_cpx],wvw_ra[ind_cpx],wvw_dec[ind_cpx],wvw_gl[ind_cpx],wvw_gb[ind_cpx],
                         wvw_cz[ind_cpx],wvw_vlsr[ind_cpx],wvw_vgsr[ind_cpx],wvw_vlg[ind_cpx],
                         wvw_vdev[ind_cpx],tmp['Flux'][ind_cpx]],
                        names=('ID','RA','Dec','l','b','cz','vlsr','vgsr','vlg','vdev','flux'))
    return wvw_cat



def get_lg(files,names=lg_colnames, data=32, col_starts=lg_colstarts):
    #script to get LG galaxy catalog from formatted catalog from McConnachie
    #Header is line 35 in emacs, Galaxies start at line 37, but with MW, skip it
    #Header also doesn't include all keywords, so maybe do that explicitly
    full_lg_cat = ascii.read(files,names=names,format='fixed_width',data_start=data,col_starts=col_starts)
    #now get the parts of the table that I actually care about
    ra = (full_lg_cat['RAh']+full_lg_cat['RAm']/60.+full_lg_cat['RAs']/3600.)*15.
    dec = full_lg_cat['Decd']+full_lg_cat['Decm']/60.+full_lg_cat['Decs']/3600.
    lg_cat = Table([full_lg_cat['Name'],ra,dec,full_lg_cat['vh']],names=('Names','RA','Dec','cz'))
    return lg_cat

def get_spring_indices(coords):
    #take a coordinate array and return indices of those that are in ALFALFA spring footprint
    ind_spring_ra = np.where((coords.ra.deg > 112.5) & (coords.ra.deg < 247.5))
    ind_spring_dec = np.where((coords.dec.deg > 0) & (coords.dec.deg < 36))
    ind_spring = np.intersect1d(ind_spring_ra,ind_spring_dec)
    return ind_spring

   
def get_neighbor_dist(coords1,vel1,coords2=None,vel2=None, Nth=3, f=0.2):
    #coords1: skycoord for primary catalog for whcih I want nearest neighbors
    #vel1: velocities for primary catalog
    #coords2,vel2: same but for catalog from which I compute distances
    #This can be a list for multiple catalogs. Defaults to None, which is input catalog
    #N: Nth nearest neihgbor
    #f: linking scale for 3D distancs
    #Note: if any values are missing (e.g., vels) need to make sure to replace iwth NaNs before using here
    #first check if I need to ignore source or not:
    if coords2 == None:
        coords2 = [coords1]
    if vel2 == None:
        vel2 = [vel1] #even if coords is different, can ignore 3d part if not given
   
    #I want to return list or array of distance values if I give multiple inputs
    #So check if I gave lists as inputs
    #if not, turn into lists
    if type(Nth) != list:
        Nth = [Nth]
    if type(f) != list:
        f = [f]
    if type(coords2) != list:
        coords2 = [coords2]
    if type(vel2) != list:
        vel2=[vel2]
    
    #then figure out how many arrays I have to return.
    #assume coords2 and vel2 match in length
    ntot = len(coords2)*len(f)*len(Nth)
    dist_array = np.zeros((ntot,len(coords1))) #make a zero array that is ntot by len of coords1
    #print dist_array.shape
    #now start the iteration and get 3D distance
    #Can always use f=0 to get 2D distance!
    #first iterate over coords2,Nth,f - different parameters
    #then calculate distance for each c in coords1
    #manually track my index for this
    variation = 0
    for c2,v2 in zip(coords2,vel2): #hopefully my zipping I implicitly check that c2,v2 have same dimensions
        for link in f:
            for n in Nth:
                #check if c2 is equivalent to c1
                #if so, keep n as is n, otherwise offset by 1 (first nearest neighbor in 0th index)
                if c2 != coords1:
                    n=n-1                
                for i,[c,v] in enumerate(zip(coords1,vel1)):
                    angsep = c.separation(c2).value
                    dv = v-v2
                    sep3d = np.sqrt( angsep**2 + (link*dv)**2)
                    sorted_sep3d = np.sort(sep3d)
                    neighbor = sorted_sep3d[n]
                    dist_array[variation,i] = neighbor
                variation = variation+1
    #print dist_array.shape
    return dist_array



def plot_Nth_neighbors(coords1,vel1,coords2=None,vel2=None,N=[3],f=[0.2],xmax=100,ymax=1.1,labels=None,cutoff=10.):
    #plots the Nth nearest neighbors
    #also takes a cutoff value, plots it and reports fraction of sources w/in that distance
    #same inputs as get_neighbor_distance
    #do same checks, because I need same things
    if coords2 == None:
        coords2 = [coords1]
    if vel2 == None:
        vel2 = [vel1] #even if coords is different, can ignore 3d part if not given
   
    #I want to return list or array of distance values if I give multiple inputs
    #So check if I gave lists as inputs
    #if not, turn into lists
    if type(N) != list:
        N = [N]
    if type(f) != list:
        f = [f]
    if type(coords2) != list:
        coords2 = [coords2]
    if type(vel2) != list:
        vel2=[vel2]
    if type(labels) != list:
        labels = [labels]
    

    #and just pass inputs directly along
    dist_array = get_neighbor_dist(coords1,vel1,coords2,vel2,N,f)

    #get labels by doing same order of iteration
    #labels=[]
    if labels== None:
        labels = []
        for cat,c2 in enumerate(coords2): 
            for link in f:
                for n in N:
                    labels.append('Cat {0}, {1} NN, f={2}'.format(cat,n,link))

    if len(labels) != len(coords2)*len(f)*len(N):
        print "Warning! length of labels doesn't match number of variations"

    #now do the plotting!
    #and get cutoff vals
    cf_cutoff = np.zeros(len(labels))
    fig, ax1=plt.subplots(1,1,figsize=(6,6))
    for i in xrange(len(labels)):
        values, bins = np.histogram(dist_array[i,:], bins=500,range=(0,100))
        cum = np.cumsum(values)/float(len(coords1))
        plt.plot(bins[:-1], cum,label=labels[i])
        plt.plot([cutoff,cutoff],[0,2],'k:')
        plt.ylim(0,ymax)
        plt.xlim(0,xmax)
        plt.legend(loc=0)
        ind = np.where(bins==cutoff)
        cf_cutoff[i] = cum[ind]



    return fig,cf_cutoff
   
