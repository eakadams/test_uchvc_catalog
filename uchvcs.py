#!/usr/bin/python

#will write functions here for dealing with things specific to uchvcs

import numpy as np
import scipy.optimize as optimization
import aplpy
from matplotlib import rc
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
from astropy.io import ascii
import colormaps as cmaps

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def shift(spec,velarr,vfit,vcen):
    #inputs are 1-d spectrum as a numpy array
    #velocity array (needed for determining fractional shift)
    #fitted central velocity
    #and velocity want to center at
    #first compute the shift:
    vshift = vfit-vcen
    #now compute the FFT of spec;
    #could probably take the simplying assumption that it is all real
    #but I'm not sure what that will to normalization later
    fft = np.fft.rfft(spec)
    #now calculate the exponent for the shift
    #first I need to figure out the velocity range covered by my array
    velrange = abs(velarr[0]-velarr[1])*len(velarr)
    #print vshift,velrange
    shift = np.arange(len(fft))*complex(0,-2*np.pi*(vshift/velrange))
    shift_fft = fft*np.exp(shift)
    inv_shift = np.fft.irfft(shift_fft)
    return inv_shift


def get_shifted_spectrum(cube,vfield,velarr,vcen,beamarea,dist=np.inf):
    #takes data cube, velocity field, velocity array, central velocity and beam area
    #need velocity array since I am not relying on header information from fits cube
    #and need beam area to convert to useful flux units
    #also note that I am assuming units of velocity field are m/s while velarray is km/s
    #in order to avoid issues of how my velfield is centered relative to full data cube
    #i am assuming that I give a trimmed data cube to this that matches velfield in shape
    #that way I can check outside that I'm taking center pixel correctly
    if cube.shape[1] != vfield.shape[0]:
        print 'Input cube does not match velocity field dimensions!'
    else:
        speclist=[]
        for i in xrange(vfield.shape[0]):
            for j in xrange(vfield.shape[1]):
                #first check for nan value
                #because then i don't have to do anything
                vfit=vfield[i,j]
                if not np.isnan(vfit):
                    #now check if i'm within sepcified distance
                    #assume this in arcsec
                    #default is inf so everyhting will pass
                    pix_dist = 4*np.sqrt( (i-vfield.shape[0]/2)**2 + (j-vfield.shape[1]/2)**2)
                    if pix_dist < dist:
                        spec = cube[:,i,j]
                        shifted_spec = shift(spec,velarr,vfit,vcen)
                        speclist.append(shifted_spec)
        specarray=np.array(speclist)
        shifted_spec = np.sum(specarray,axis=0)/beamarea
        nbeam = len(specarray)/beamarea
        return shifted_spec,nbeam
        
def gauss(x,a,b,c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def doublegauss(x,a,b,c,d,e,f):
    return gauss(x,a,b,c)+gauss(x,d,e,f)


def fit_gauss(velarr,spec,rms=2e-3,guess=[0.4,50,10]):
    #take velarr and spec as required inputs
    #use rms to set error (although maybe i need to worry about units?)
    #i probably need to know how many beams there are in my integration area to properly calculate rms
    #will leave this for future and just use for now
    error = np.zeros(len(spec))+rms
    fit = optimization.curve_fit(gauss,velarr,spec,guess,error)
    redchisq=get_redchisq(velarr,spec,error,fit,type='gauss')
    return fit[0][0],fit[0][1],fit[0][2],redchisq


def fit_doublegauss(velarr,spec,rms=2e-3,guess=[0.4,50,10,0.1,50,3]):
    #take velarr and spec as required inputs
    #use rms to set error (although maybe i need to worry about units?)
    #i probably need to know how many beams there are in my integration area to properly calculate rms
    #will leave this for future and just use for now
    error = np.zeros(len(spec))+rms
    fit = optimization.curve_fit(doublegauss,velarr,spec,guess,error)
    redchisq=get_redchisq(velarr,spec,error,fit,type='doublegauss')
    return fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],redchisq



def get_redchisq(velarr,spec,error,fit,type='None'):
    #this takes velocity array, spectrum and the fit
    #and calculates the reduced chi squared
    #first get an array of fitted values
    #to do this need to know what type of fit to use - 'gauss' or 'doublegauss'
    if type == 'gauss': 
        fitvals = gauss(velarr,fit[0][0],fit[0][1],fit[0][2])
        chisq = sum( ((fitvals-spec)/error)**2 )
        redchisq = chisq / (len(spec)-3)  #divide by dofs
        return redchisq
    elif type=='doublegauss':
        fitvals = doublegauss(velarr,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5])
        chisq = sum( ((fitvals-spec)/error)**2 )
        redchisq = chisq / (len(spec)-6)  #divide by dofs
        return redchisq
    else:
        print 'Improper type of fit given'

 
def get_pvarrow(ra,dec,pa,length):
    #take center, pa and length to get params of pv arrow for aplpy plotting
    dx = -np.sin(pa*np.pi/180.)*length/60. #in degrees!
    dy = -np.cos(pa*np.pi/180.)*length/60. #in degrees!
    xs=ra-dx/2.
    ys=dec - dy/2.
    return xs,ys,dx,dy

def get_mom_map_figure(nhi_210,mom1_210,mom2_210,nhi_100,mom1_100,mom2_100,min_vf,max_vf,min_vd,max_vd,
                       levs210,levs100,levs_vel,ra,dec,pa,length,mask210=None,mask100=None,
                       lbw=False,ra_alf = 0.,dec_alf=0.,fov=0.5):
    #take moment maps and some plotting params to make figures for WSRT UCHVC paper automatically
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xs,ys,dx,dy=get_pvarrow(ra,dec,pa,length)

    fig=plt.figure(figsize=(12,10))
    f1 = aplpy.FITSFigure(mom1_210,figure=fig,subplot=[0.08,0.55,0.35,0.45])
    f1.recenter(ra,dec, radius=fov/2.)
    f1.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f1.show_contour(nhi_210,levels=levs210,colors='black')
    f1.add_colorbar()
    f1.colorbar.show()
    f1.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f1.colorbar.set_axis_label_font(size=12)
    f1.tick_labels.set_xformat('ddd.dd')
    f1.tick_labels.set_yformat('ddd.dd')
    f1.axis_labels.set_font(size=15)
    f1.add_beam()
    f1.beam.hide()
    f1.beam.set_edgecolor('black')
    f1.beam.set_facecolor('none')
    f1.beam.show(facecolor='none')
    f1.show_arrows(xs,ys,dx,dy,color='black',width=.1,head_width=1)
    if mask210 != None:
        f1.show_contour(mask210,levels=[0.5],colors='Gray',linewidths=3)
    if lbw == True:
        f1.show_circles(ra_alf,dec_alf,3.5/(2.*60.),color='indigo',linewidth=2)

    f2 = aplpy.FITSFigure(mom2_210,figure=fig,subplot=[0.57,0.55,0.35,0.45])
    f2.recenter(ra,dec, radius=fov/2.)
    f2.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f2.show_contour(mom1_210,levels=levs_vel,colors='black')
    f2.add_colorbar()
    f2.colorbar.show()
    f2.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f2.colorbar.set_axis_label_font(size=12)
    f2.tick_labels.set_xformat('ddd.dd')
    f2.tick_labels.set_yformat('ddd.dd')
    f2.show_arrows(xs,ys,dx,dy,color='black',width=.1,head_width=1)
    f2.axis_labels.set_font(size=15)

    f3 = aplpy.FITSFigure(mom1_100,figure=fig,subplot=[0.08,0.05,0.35,0.45])
    f3.recenter(ra,dec, radius=fov/2.)
    f3.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f3.show_contour(nhi_100,levels=levs100,colors='black')
    f3.add_colorbar()
    f3.colorbar.show()
    f3.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f3.colorbar.set_axis_label_font(size=12)
    f3.tick_labels.set_xformat('ddd.dd')
    f3.tick_labels.set_yformat('ddd.dd')
    f3.add_beam()
    f3.beam.hide()
    f3.beam.set_edgecolor('black')
    f3.beam.set_facecolor('none')
    f3.beam.show(facecolor='none')
    f3.axis_labels.set_font(size=15)
    f3.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    if mask100 != None:
        f3.show_contour(mask100,levels=[0.5],colors='Gray',linewidths=3)  
    if lbw == True:
        f3.show_circles(ra_alf,dec_alf,3.5/(2.*60.),color='indigo',linewidth=3)


    f4 = aplpy.FITSFigure(mom2_100,figure=fig,subplot=[0.57,0.05,0.35,0.45])
    f4.recenter(ra,dec, radius=fov/2.)
    f4.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f4.show_contour(mom1_100,levels=levs_vel,colors='black')
    f4.add_colorbar()
    f4.colorbar.show()
    f4.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f4.colorbar.set_axis_label_font(size=12)
    f4.tick_labels.set_xformat('ddd.dd')
    f4.tick_labels.set_yformat('ddd.dd')
    f4.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    f4.axis_labels.set_font(size=15)

    #plt.savefig(name)

    return fig


def get_three_mom_map_figure(nhi_210,mom1_210,mom2_210,nhi_100,mom1_100,mom2_100,nhi_60,mom1_60,mom2_60,
                       min_vf,max_vf,min_vd,max_vd,
                       levs210,levs100,levs60,levs_vel,ra,dec,pa,length,mask210=None,mask100=None,mask60=None,
                       lbw=False,ra_alf = 0.,dec_alf=0.,fov=0.5):
    #take moment maps and some plotting params to make figures for WSRT UCHVC paper automatically
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xs,ys,dx,dy=get_pvarrow(ra,dec,pa,length)

    fig=plt.figure(figsize=(12,15))
    f1 = aplpy.FITSFigure(mom1_210,figure=fig,subplot=[0.08,0.7,0.35,0.3])
    f1.recenter(ra,dec, radius=fov/2.)
    f1.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f1.show_contour(nhi_210,levels=levs210,colors='black')
    f1.add_colorbar()
    f1.colorbar.show()
    f1.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f1.colorbar.set_axis_label_font(size=12)
    f1.tick_labels.set_xformat('ddd.dd')
    f1.tick_labels.set_yformat('ddd.dd')
    f1.axis_labels.set_font(size=15)
    f1.add_beam()
    f1.beam.hide()
    f1.beam.set_edgecolor('black')
    f1.beam.set_facecolor('none')
    f1.beam.show(facecolor='none')
    f1.show_arrows(xs,ys,dx,dy,color='black',width=.1,head_width=1)
    if mask210 != None:
        f1.show_contour(mask210,levels=[0.5],colors='Gray',linewidths=3)
    if lbw == True:
        f1.show_circles(ra_alf,dec_alf,3.5/(2.*60.),color='indigo',linewidth=2)

    f2 = aplpy.FITSFigure(mom2_210,figure=fig,subplot=[0.57,0.7,0.35,0.3])
    f2.recenter(ra,dec, radius=fov/2.)
    f2.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f2.show_contour(mom1_210,levels=levs_vel,colors='black')
    f2.add_colorbar()
    f2.colorbar.show()
    f2.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f2.colorbar.set_axis_label_font(size=12)
    f2.tick_labels.set_xformat('ddd.dd')
    f2.tick_labels.set_yformat('ddd.dd')
    f2.show_arrows(xs,ys,dx,dy,color='black',width=.1,head_width=1)
    f2.axis_labels.set_font(size=15)

    f3 = aplpy.FITSFigure(mom1_100,figure=fig,subplot=[0.08,0.37,0.35,0.3])
    f3.recenter(ra,dec, radius=fov/2.)
    f3.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f3.show_contour(nhi_100,levels=levs100,colors='black')
    f3.add_colorbar()
    f3.colorbar.show()
    f3.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f3.colorbar.set_axis_label_font(size=12)
    f3.tick_labels.set_xformat('ddd.dd')
    f3.tick_labels.set_yformat('ddd.dd')
    f3.add_beam()
    f3.beam.hide()
    f3.beam.set_edgecolor('black')
    f3.beam.set_facecolor('none')
    f3.beam.show(facecolor='none')
    f3.axis_labels.set_font(size=15)
    f3.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    if mask100 != None:
        f3.show_contour(mask100,levels=[0.5],colors='Gray',linewidths=3)  
    if lbw == True:
        f3.show_circles(ra_alf,dec_alf,3.5/(2.*60.),color='indigo',linewidth=3)


    f4 = aplpy.FITSFigure(mom2_100,figure=fig,subplot=[0.57,0.37,0.35,0.3])
    f4.recenter(ra,dec, radius=fov/2.)
    f4.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f4.show_contour(mom1_100,levels=levs_vel,colors='black')
    f4.add_colorbar()
    f4.colorbar.show()
    f4.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f4.colorbar.set_axis_label_font(size=12)
    f4.tick_labels.set_xformat('ddd.dd')
    f4.tick_labels.set_yformat('ddd.dd')
    f4.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    f4.axis_labels.set_font(size=15)

    f5 = aplpy.FITSFigure(mom1_60,figure=fig,subplot=[0.08,0.03,0.35,0.3])
    f5.recenter(ra,dec, radius=fov/2.)
    f5.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f5.show_contour(nhi_60,levels=levs60,colors='black')
    f5.add_colorbar()
    f5.colorbar.show()
    f5.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f5.colorbar.set_axis_label_font(size=12)
    f5.tick_labels.set_xformat('ddd.dd')
    f5.tick_labels.set_yformat('ddd.dd')
    f5.add_beam()
    f5.beam.hide()
    f5.beam.set_edgecolor('black')
    f5.beam.set_facecolor('none')
    f5.beam.show(facecolor='none')
    f5.axis_labels.set_font(size=15)
    f5.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    if mask60 != None:
        f5.show_contour(mask60,levels=[0.5],colors='Gray',linewidths=3)  
    if lbw == True:
        f5.show_circles(ra_alf,dec_alf,3.5/(2.*60.),color='indigo',linewidth=3)


    f6 = aplpy.FITSFigure(mom2_60,figure=fig,subplot=[0.57,0.03,0.35,0.3])
    f6.recenter(ra,dec, radius=fov/2.)
    f6.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f6.show_contour(mom1_60,levels=levs_vel,colors='black')
    f6.add_colorbar()
    f6.colorbar.show()
    f6.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f6.colorbar.set_axis_label_font(size=12)
    f6.tick_labels.set_xformat('ddd.dd')
    f6.tick_labels.set_yformat('ddd.dd')
    f6.show_arrows(xs,ys,dx,dy,color='black',width=.2,head_width=2)
    f6.axis_labels.set_font(size=15)

    #plt.savefig(name)

    return fig


def get_one_mom_map_figure(nhi_210,mom1_210,mom2_210,min_vf,max_vf,min_vd,max_vd,levs210,levs_vel,ra,dec,pa,length,mask=None,fov=0.5):
    #take moment maps and some plotting params to make figures for WSRT UCHVC paper automatically
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xs,ys,dx,dy=get_pvarrow(ra,dec,pa,length)

    fig=plt.figure(figsize=(12,5))
    f1 = aplpy.FITSFigure(mom1_210,figure=fig,subplot=[0.08,0.075,0.35,0.9])
    f1.recenter(ra,dec, radius=fov/2.)
    f1.show_colorscale(cmap=cmaps.viridis,vmin=min_vf,vmax=max_vf)
    f1.show_contour(nhi_210,levels=levs210,colors='black')
    f1.add_colorbar()
    f1.colorbar.show()
    f1.colorbar.set_axis_label_text(r'cz (km s$^{-1}$)')
    f1.colorbar.set_axis_label_font(size=12)
    f1.tick_labels.set_xformat('ddd.dd')
    f1.tick_labels.set_yformat('ddd.dd')
    f1.add_beam()
    f1.beam.hide()
    f1.beam.set_edgecolor('black')
    f1.beam.set_facecolor('none')
    f1.beam.show(facecolor='none')
    f1.axis_labels.set_font(size=15)
    f1.show_arrows(xs,ys,dx,dy,color='black',width=.4,head_width=4)
    if mask != None:
        f1.show_contour(mask,levels=[0.5],colors='Gray',linewidths=3)

    f2 = aplpy.FITSFigure(mom2_210,figure=fig,subplot=[0.57,0.075,0.35,0.9])
    f2.recenter(ra,dec, radius=fov/2.)
    f2.show_colorscale(cmap=cmaps.viridis,vmin=min_vd,vmax=max_vd)
    f2.show_contour(mom1_210,levels=levs_vel,colors='black')
    f2.add_colorbar()
    f2.colorbar.show()
    f2.colorbar.set_axis_label_text(r'$\sigma$ (km s$^{-1}$)')
    f2.colorbar.set_axis_label_font(size=12)
    f2.tick_labels.set_xformat('ddd.dd')
    f2.tick_labels.set_yformat('ddd.dd')
    f2.show_arrows(xs,ys,dx,dy,color='black',width=.4,head_width=4)
    f2.axis_labels.set_font(size=15)

    return fig


def get_pvslice_figure(pv210,pv100,rms210,rms100,cenvel,minvel,maxvel,velvals,length,ticksep=2.,
                       showvel=False,extent=False):
    #get the pv slice figure
    #first define the offset ticks I wants in arcmin
    nticks=length/ticksep + 1 #plus one is for center '0' slice
    tickvals = np.arange(nticks)*ticksep - length/2
    offsetticklabels=map(str, tickvals)

    #so I have the tick values I want but I'm not sure how those map onto pixls yet
    #lets get relevant header value from pv slice
    pv210_header=fits.getheader(pv210)
    pv100_header=fits.getheader(pv100)
    if pv210_header['cunit1'] == 'arcsec':
        offsettickvals = (tickvals*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        #put tickvals into arcsec, offset by reference value, then into pix vals, then move to reference pixel
    
    if pv100_header['cunit1'] == 'arcsec':
        offsettickvals100 = (tickvals*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']

    if pv210_header['cunit2']=='m/s':
        veltickvals = (velvals*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        cenvel_pix = (cenvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        maxvel_pix = (maxvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        minvel_pix = (minvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        #assume 210 and 100 data are on same vel scale

    if extent != False:
        negextent210 = ((-extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        posextent210 = ((extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        negextent100 = ((-extent/2.)*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']
        posextent100 = ((extent/2.)*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']

    x1_210 = pv210_header['NAXIS1']-0.5
    x1_100 = pv100_header['NAXIS1']-0.5
    pvlevs210 = np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms210
    #(np.append(-2,np.arange(20)*2+2))*rms210
    pvlevs100=np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms100
    #(np.append(-2,np.arange(20)*2+2))*rms100
    velticklabels=map(str,velvals)

    fig, (ax1,ax2)=plt.subplots(2,1,figsize=(6,12))

    pv210im = fits.getdata(pv210)
    pv100im = fits.getdata(pv100)

    im=ax1.imshow(pv210im,cmap='Oranges',interpolation='nearest')
    ax1.set_aspect('auto',adjustable='box')
    ax1.set_xticks(offsettickvals)
    ax1.set_xticklabels(offsetticklabels)
    ax1.set_xlabel('Offset (arcmin)',size=15)
    ax1.set_yticks(veltickvals)
    ax1.set_yticklabels(velticklabels)
    ax1.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax1.contour(pv210im,levels=pvlevs210,colors='black')
    if showvel==True:
        ax1.plot([-0.5,x1_210],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax1.plot([-0.5,x1_210],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax1.plot([-0.5,x1_210],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax1.plot([negextent210,negextent210],[0,24],'k--',linewidth=2)
        ax1.plot([posextent210,posextent210],[0,24],'k--',linewidth=2)


    im=ax2.imshow(pv100im,cmap='Oranges',interpolation='nearest')
    ax2.set_aspect('auto',adjustable='box')
    ax2.set_xticks(offsettickvals100)
    ax2.set_xticklabels(offsetticklabels)
    ax2.set_xlabel('Offset (arcmin)',size=15)
    ax2.set_yticks(veltickvals)
    ax2.set_yticklabels(velticklabels)
    ax2.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax2.contour(pv100im,levels=pvlevs100,colors='black',interpolation='nearest')
    if showvel == True:
        ax2.plot([-0.5,x1_100],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax2.plot([-0.5,x1_100],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax2.plot([-0.5,x1_100],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax2.plot([negextent100,negextent100],[0,24],'k--',linewidth=2)
        ax2.plot([posextent100,posextent100],[0,24],'k--',linewidth=2)


    return fig



def get_three_pvslice_figure(pv210,pv100,pv60,rms210,rms100,rms60,cenvel,minvel,maxvel,velvals,length,ticksep=2.,
                       showvel=False,extent=False):
    #get the pv slice figure
    #first define the offset ticks I wants in arcmin
    nticks=length/ticksep + 1 #plus one is for center '0' slice
    tickvals = np.arange(nticks)*ticksep - length/2
    offsetticklabels=map(str, tickvals)

    #so I have the tick values I want but I'm not sure how those map onto pixls yet
    #lets get relevant header value from pv slice
    pv210_header=fits.getheader(pv210)
    pv100_header=fits.getheader(pv100)
    pv60_header=fits.getheader(pv60)
    if pv210_header['cunit1'] == 'arcsec':
        offsettickvals = (tickvals*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        #put tickvals into arcsec, offset by reference value, then into pix vals, then move to reference pixel
    
    if pv100_header['cunit1'] == 'arcsec':
        offsettickvals100 = (tickvals*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']

    if pv60_header['cunit1'] == 'arcsec':
        offsettickvals60 = (tickvals*60.-pv60_header['crval1'])/pv60_header['cdelt1'] + pv60_header['crpix1']

    if pv210_header['cunit2']=='m/s':
        veltickvals = (velvals*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        cenvel_pix = (cenvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        maxvel_pix = (maxvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        minvel_pix = (minvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        #assume 210 and 100 data are on same vel scale

    if extent != False:
        negextent210 = ((-extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        posextent210 = ((extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        negextent100 = ((-extent/2.)*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']
        posextent100 = ((extent/2.)*60.-pv100_header['crval1'])/pv100_header['cdelt1'] + pv100_header['crpix1']
        negextent60 = ((-extent/2.)*60.-pv60_header['crval1'])/pv60_header['cdelt1'] + pv60_header['crpix1']
        posextent60 = ((extent/2.)*60.-pv60_header['crval1'])/pv60_header['cdelt1'] + pv60_header['crpix1']

    x1_210 = pv210_header['NAXIS1']-0.5
    x1_100 = pv100_header['NAXIS1']-0.5
    x1_60 = pv60_header['NAXIS1']-0.5
    pvlevs210 = np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms210
    #(np.append(-2,np.arange(20)*2+2))*rms210
    pvlevs100=np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms100
    pvlevs60=np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms60
    #(np.append(-2,np.arange(20)*2+2))*rms100
    velticklabels=map(str,velvals)

    fig, (ax1,ax2,ax3)=plt.subplots(3,1,figsize=(6,18))
    #fig=plt.figure(figsize=(6,18))
    #ax1=plt.axes([0.16,0.7,0.8,0.3])
    #ax2=plt.axes([0.16,0.37,0.8,0.3])
    #ax3=plt.axes([0.16,0.03,0.8,0.3])

    pv210im = fits.getdata(pv210)
    pv100im = fits.getdata(pv100)
    pv60im = fits.getdata(pv60)

    im=ax1.imshow(pv210im,cmap='Oranges',interpolation='nearest')
    ax1.set_aspect('auto',adjustable='box')
    ax1.set_xticks(offsettickvals)
    ax1.set_xticklabels(offsetticklabels)
    ax1.set_xlabel('Offset (arcmin)',size=15)
    ax1.set_yticks(veltickvals)
    ax1.set_yticklabels(velticklabels)
    ax1.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax1.contour(pv210im,levels=pvlevs210,colors='black')
    if showvel==True:
        ax1.plot([-0.5,x1_210],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax1.plot([-0.5,x1_210],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax1.plot([-0.5,x1_210],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax1.plot([negextent210,negextent210],[0,24],'k--',linewidth=2)
        ax1.plot([posextent210,posextent210],[0,24],'k--',linewidth=2)


    im=ax2.imshow(pv100im,cmap='Oranges',interpolation='nearest')
    ax2.set_aspect('auto',adjustable='box')
    ax2.set_xticks(offsettickvals100)
    ax2.set_xticklabels(offsetticklabels)
    ax2.set_xlabel('Offset (arcmin)',size=15)
    ax2.set_yticks(veltickvals)
    ax2.set_yticklabels(velticklabels)
    ax2.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax2.contour(pv100im,levels=pvlevs100,colors='black',interpolation='nearest')
    if showvel == True:
        ax2.plot([-0.5,x1_100],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax2.plot([-0.5,x1_100],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax2.plot([-0.5,x1_100],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax2.plot([negextent100,negextent100],[0,24],'k--',linewidth=2)
        ax2.plot([posextent100,posextent100],[0,24],'k--',linewidth=2)


    im=ax3.imshow(pv60im,cmap='Oranges',interpolation='nearest')
    ax3.set_aspect('auto',adjustable='box')
    ax3.set_xticks(offsettickvals60)
    ax3.set_xticklabels(offsetticklabels)
    ax3.set_xlabel('Offset (arcmin)',size=15)
    ax3.set_yticks(veltickvals)
    ax3.set_yticklabels(velticklabels)
    ax3.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax3.contour(pv60im,levels=pvlevs60,colors='black',interpolation='nearest')
    if showvel == True:
        ax3.plot([-0.5,x1_60],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax3.plot([-0.5,x1_60],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax3.plot([-0.5,x1_60],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax3.plot([negextent60,negextent60],[0,24],'k--',linewidth=2)
        ax3.plot([posextent60,posextent60],[0,24],'k--',linewidth=2)


    return fig


def get_one_pvslice_figure(pv210,rms210,cenvel,minvel,maxvel,velvals,length,ticksep=2.,showvel=False,extent=False):
    #get the pv slice figure
    #first define the offset ticks I wants in arcmin
    nticks=length/ticksep + 1 #plus one is for center '0' slice
    tickvals = np.arange(nticks)*ticksep - length/2
    offsetticklabels=map(str, tickvals)

    #so I have the tick values I want but I'm not sure how those map onto pixls yet
    #lets get relevant header value from pv slice
    pv210_header=fits.getheader(pv210)
    if pv210_header['cunit1'] == 'arcsec':
        offsettickvals = (tickvals*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        #put tickvals into arcsec, offset by reference value, then into pix vals, then move to reference pixel
    

    if pv210_header['cunit2']=='m/s':
        veltickvals = (velvals*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        cenvel_pix = (cenvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        maxvel_pix = (maxvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        minvel_pix = (minvel*1000.-pv210_header['crval2'])/pv210_header['cdelt2'] + pv210_header['crpix2']
        #assume 210 and 100 data are on same vel scale

    if extent != False:
        negextent210 = ((-extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']
        posextent210 = ((extent/2.)*60.-pv210_header['crval1'])/pv210_header['cdelt1'] + pv210_header['crpix1']


    x1_210 = pv210_header['NAXIS1']-0.5
    pvlevs210 = np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*rms210
    #(np.append(-2,np.arange(20)*2+2))*rms210
    #(np.append(-2,np.arange(20)*2+2))*rms100
    velticklabels=map(str,velvals)

    fig, ax1=plt.subplots(1,1,figsize=(6,6))

    pv210im = fits.getdata(pv210)

    im=ax1.imshow(pv210im,cmap='Oranges',interpolation='nearest')
    ax1.set_aspect('auto',adjustable='box')
    ax1.set_xticks(offsettickvals)
    ax1.set_xticklabels(offsetticklabels)
    ax1.set_xlabel('Offset (arcmin)',size=15)
    ax1.set_yticks(veltickvals)
    ax1.set_yticklabels(velticklabels)
    ax1.set_ylabel(r'Velocity (km s$^{-1}$)',size=15)
    ax1.contour(pv210im,levels=pvlevs210,colors='black')
    if showvel==True:
        ax1.plot([-0.5,x1_210],[cenvel_pix,cenvel_pix],'k:',linewidth=2)
        ax1.plot([-0.5,x1_210],[maxvel_pix,maxvel_pix],'k',linewidth=2)
        ax1.plot([-0.5,x1_210],[minvel_pix,minvel_pix],'k',linewidth=2)
    if extent != False:
        ax1.plot([negextent210,negextent210],[0,24],'k--',linewidth=2)
        ax1.plot([posextent210,posextent210],[0,24],'k--',linewidth=2)



    return fig


def get_alfalfa_overlay(alfalfa,wsrt210,wsrt_rms,ara,adec,wra,wdec,a,b,pa,rhalf,bx,by,bw,bh,vmin=None,vmax=None,fov=None):
    a_deg=a/60.
    b_deg=b/60.
    abar = np.sqrt(a_deg*b_deg)
    r_deg = rhalf/60.
    rms_levs = np.array([-4,-2*np.sqrt(2),-2,2,2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64])*wsrt_rms
    #print rms_levs
    fig=plt.figure(figsize=(8,8))
    f1=aplpy.FITSFigure(alfalfa,figure=fig,subplot=[0.15, 0.1, 0.8, 0.8])
    if fov ==None:
        f1.recenter(ara,adec, radius=0.25)
    else:
        f1.recenter(ara,adec, radius=fov/2.)
    f1.axis_labels.set_font(size=15)
    f1.show_grayscale(invert=True,vmin=vmin,vmax=vmax)
    f1.show_rectangles(bx,by,bw,bh,edgecolor=cmaps.viridis(0.0),linestyle='--',linewidth=3)
    if wsrt210 != 'none':
        f1.show_contour(wsrt210,levels=rms_levs,colors=cmaps.viridis(0.99),linewidth=5)
    f1.show_ellipses(ara,adec,a_deg,b_deg,angle=pa,edgecolor=cmaps.viridis(0.33),linewidth=5)
    f1.show_circles(ara,adec,abar/2.,edgecolor=cmaps.viridis(0.33),linestyle='--',linewidth=5)
    f1.show_markers(ara,adec,c=cmaps.viridis(0.33),marker='x',s=50,linewidths=4)
    f1.show_circles(wra,wdec,r_deg,edgecolor=cmaps.viridis(0.66),linestyle=':',linewidth=10)
    if wsrt210 != 'none':
        f1.show_markers(wra,wdec,c=cmaps.viridis(0.66),marker='+',s=50,linewidths=4)

    return fig


def get_spec_figure(alfalfa,lbw,wsrt210,wsrt100,noise210,noise100,ymin=-0.01,ymax=0.2,xcen=0,xspan=300,chlo=0,chhi=0,extraspec=None,wsrt60=None,noise60=None,name=None):
    alfa = ascii.read(alfalfa)
    if extraspec != None:
        alfa2 = ascii.read(extraspec)
    if lbw != 'none':
        lb = ascii.read(lbw)
    if wsrt60 != None:
        w60 = ascii.read(wsrt60)
    if noise60 != None:
        n60 = ascii.read(noise60)
    w210 = ascii.read(wsrt210)
    w100=ascii.read(wsrt100)
    n210 = ascii.read(noise210)
    n100=ascii.read(noise100)
    fig=plt.figure(figsize=(10,6))
    gs=gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(alfa['cz'],alfa['flux']/1000.,c=cmaps.viridis(0.0),label='ALFALFA',linewidth=2)
    if lbw != 'none':
        ax0.plot(lb['col2'],lb['col6']/1000.,'k:',linewidth=2,label='LBW')
    if extraspec != None:
        ax0.plot(alfa2['cz'],alfa2['flux']/1000.,'k--',label='ALFALFA')
    if wsrt60 != None:
        ax0.plot(w60['vel'],w60['flux'],c=cmaps.viridis(0.75),linestyle='-.',
                 label='WSRT 60"',linewidth=4)
    ax0.plot(w210['vel'],w210['flux'],c=cmaps.viridis(0.25),label='WSRT 210"',
             linewidth=2,linestyle='--')
    ax0.plot(w100['vel'],w100['flux'],c=cmaps.viridis(0.5),linestyle=':',
             linewidth=4,label='WSRT 105"')
    xmin = xcen-xspan/2.
    xmax=xcen+xspan/2.
    ax0.axis([xmin,xmax,ymin,ymax])
    ax0.plot([w210['vel'][chlo],w210['vel'][chlo]],[-1,10],'k:')
    ax0.plot([w210['vel'][chhi],w210['vel'][chhi]],[-1,10],'k:')

    ax0.set_ylabel('Flux (Jy)',size=15)
    ax0.set_xlabel(r'v$_{hel}$ (km s$^{-1}$)',size=15)

    ax0.legend(loc=1)
            
    if name != None:
        ax0.text(0.05,0.8,name,fontsize=15,transform=ax0.transAxes)

    nmax = (ymax-ymin)/6.
    nmin = -(ymax-ymin)/6.

    ax1 = plt.subplot(gs[1])
    ax1.plot(n210['vel'], n210['flux'],c=cmaps.viridis(0.25),linewidth=2,linestyle='--')
    ax1.axis([xmin,xmax,nmin,nmax])
    if noise60 != None:
        ax1.plot(n60['vel'],n60['flux'],c=cmaps.viridis(0.75),linestyle='-.',linewidth=4)
    ax1.plot(n100['vel'], n100['flux'],c=cmaps.viridis(0.5),linewidth=4,linestyle=':')
    ax1.plot([w210['vel'][chlo],w210['vel'][chlo]],[-1,10],'k:')
    ax1.plot([w210['vel'][chhi],w210['vel'][chhi]],[-1,10],'k:')
            
    return fig

def get_spec_vla(alfalfa,vlarob,vlatap,vlarobnoise,vlatapnoise,ymin=-0.01,ymax=0.2,xcen=0,xspan=300,chlo=0,chhi=0,lbw=None):
    alfa = ascii.read(alfalfa)
    if lbw != None:
        lb = ascii.read(lbw)
    rob = ascii.read(vlarob)
    tap=ascii.read(vlatap)
    nrob = ascii.read(vlarobnoise)
    ntap=ascii.read(vlatapnoise)
    fig=plt.figure(figsize=(10,6))
    gs=gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.plot(alfa['cz'],alfa['flux']/1000.,'k',label='ALFALFA')
    if lbw != None:
        ax0.plot(lb['vel'],lb['flux'],'k--',linewidth=2,label='LBW')
    ax0.plot(rob['vel'],rob['flux'],'r',label='VLA Robust')
    ax0.plot(tap['vel'],tap['flux'],'r--',linewidth=2,label='VLA Tapered')
    xmin = xcen-xspan/2.
    xmax=xcen+xspan/2.
    ax0.axis([xmin,xmax,ymin,ymax])
    ax0.plot([rob['vel'][chlo],rob['vel'][chlo]],[-1,10],'k:')
    ax0.plot([rob['vel'][chhi],rob['vel'][chhi]],[-1,10],'k:')

    ax0.set_ylabel('Flux (Jy)',size=15)
    ax0.set_xlabel(r'v$_{hel}$ (km s$^{-1}$)',size=15)

    ax0.legend(loc=1)
            
    nmax = (ymax-ymin)/6.
    nmin = -(ymax-ymin)/6.

    ax1 = plt.subplot(gs[1])
    ax1.plot(nrob['vel'], nrob['flux'],'r')
    ax1.axis([xmin,xmax,nmin,nmax])
    ax1.plot(ntap['vel'], ntap['flux'],'r--',linewidth=2)
    ax1.plot([rob['vel'][chlo],rob['vel'][chlo]],[-1,10],'k:')
    ax1.plot([rob['vel'][chhi],rob['vel'][chhi]],[-1,10],'k:')
            
    return fig


def get_vdev(l,b,vlsr):
    rsun = 8.5
    rgal=26.
    thick=4.
    rflat=0.5
    vflat=220.
    z1=1
    z2=3
    #do a calculation based on info in book by woerden (get details)
    #l,b=get_lb(filename)
    #vlsr=get_vlsr(filename)
    vdev=np.zeros(len(vlsr))
    for i,(v,li,bi) in enumerate(zip(vlsr,l,b)):
        #define arrays for the distance along los plus radius and z at that distance
        #use an arbitrary range for distance to start; will have to calculate max distance and cutoff in next step
        darr=np.arange(0,20,0.1)
        rarr = rsun * np.sqrt( np.cos(bi*np.pi/180.)**2 * (darr/rsun)**2 - np.cos(bi*np.pi/180.)*np.cos(li*np.pi/180.)*(darr/rsun) +1 )
        zarr=darr*np.sin(bi*np.pi/180.)
        #now find the index at which i have to truncate arrays
        #this happens if r>rmax or abs(z)>zmax
        #first calculate zmax based on rarr
        #need to account for warp for points exterior to sun
        rext=np.where(rarr>=rgal)
        zmax=np.zeros(len(darr))
        zmax.fill(z1)
        zmax[rext] = z1+(z2-z1)*(rarr[rext]/rsun-1)**2/4
        #now check for where abs(z) gt zmax and r > rmax
        #do this by iterating through array manually
        #not the smartest way but the easiest to code
        #do in a while loop to let me break out
        ind=-1
        while ind < 0:
            for j,(d,r,z,zm) in enumerate(zip(darr,rarr,zarr,zmax)):
                if r > rgal:
                    ind=j
                if abs(z)>zm:
                    ind=j
        #now truncate the arrays at ind
        #really only care about radius values
        rvals=rarr[0:ind]
        #now calculate the velocity array based on this
        #i am assuming that i am only probing flat part of rotation curve since i don't think for my purposes that line of sight will ever cross r=0.5
        #if i wanted to generalize this coe, I would have to worry about this case
        varr = (rsun/rvals -1)*vflat*np.sin(li*np.pi/180.)*np.cos(bi*np.pi/180.)
        #now calculate vdev based on vel value
        if v<0:
            vdev[i] = v-min(varr)
            if vdev[i] > 0:
                vdev[i] = 0
        else:
            vdev[i] = v-max(varr)
            if vdev[i] < 0:
                vdev[i] = 0
       
    
    return vdev
    
def get_Nth_neighbor(coords1,vel1,coords2=None,vel2=None, N=3, f=0.2):
    #coords1: skycoord for primary catalog for whcih I want nearest neighbors
    #vel1: velocities for primary catalog
    #coords2,vel2: same but for catalog from which I compute distances
    #If none is given, default to primary (w/in self)
    #N: Nth nearest neihgbor
    #f: linking scale for 3D distancs
    #Note: if any values are missing (e.g., vels) need to make sure to replace iwth NaNs before using here
    #first check if I need to ignore source or not:
    if coords2 == None:
        ind_neighbor = N
    else:
        ind_neighbor = N-1
    if coords2 == None:
        coords2 = coords1
    if vel2 == None:
        vel2 = vel1 #even if coords is different, can ignore 3d part if not given
   
    dist2d = np.zeros(len(coords1))
    dist3d = np.zeros(len(coords1))
    for i,c in enumerate(coords1):
        sep = coords2.separation(coords1[i])
        angsep = np.array(sep.value)
        dv = vel1 - vel2
        dist2d[i] = np.sort(angsep)[ind_neighbor]
        sep3d = np.sqrt((angsep)**2 + (f*dv)**2)
        dist3d[i] = np.sort(sep3d)[ind_neighbor]

    return dist2d,dist3d


def plot_Nth_neighbors(coords1,vel1,coords2=None,vel2=None,N=[3],f=[0.2]):
    #plots the Nth nearest neighbors
    #same inputs as get_Nth_neighbor
    #Except! coords2, vel2, N, f can all be arrays
    #Which means they must be passed as lists!
    #in practice, it will be easiest to only iterate on one at a time
    if coords2 == None:
        N=np.array(N)+1 #since I won't subtract from it when calling get_Nth_neighbors because I'll pass c2
#    else:
#        ind_neighbor = np.array(N)-1
    if coords2 == None:
        coords2 = [coords1]
    if vel2 == None:
        vel2 = vel1 #even if coords is different, can ignore 3d part if not given
    #now I will have to be fancy with my iteration, am assuming I can iterate over almost everything
    #Not sure if there's an order that makes most sense - top down or bottom up?
    #Will go top down and just try and make sure I only ever call this function in a reasonable manner
    ntot = len(coords2)*len(N)*len(f)
    dist2d = []#np.empty(ntot,len(coords1))
    dist3d = []#np.empty(ntot,len(coords1))
    label = [] #np.empty(ntot)
    i=0 #index
    for k,[cat2,v2] in enumerate(zip(coords2,vel2)):
        for neigh in N:
            for link in f: #only iterate for 3D
                d2d,d3d = get_Nth_neighbor(coords1,vel1,cat2,v2, neigh, link)
                dist2d.append(d2d) #[i,:] = d2d
                dist3d.append (d3d) #[i,:] = d3d
                label.append('Cat '+str(k)+', '+str(neigh)+'th N w/ '+str(link)+'deg/km/s')
                #i=i+1
    #now do the plotting!
    fig, ax1=plt.subplots(1,1,figsize=(6,6))
    for i in xrange(len(label)):
        values2d, bins = np.histogram(dist2d[i], bins=500,range=(0,100))
        cum2d = np.cumsum(values2d)/float(len(coords1))
        values3d, bins = np.histogram(dist3d[i], bins=500,range=(0,100))
        cum3d = np.cumsum(values3d)/float(len(coords1))
        plt.plot(bins[:-1], cum2d,linestyle='--',label='2D, '+label[i])
        plt.plot(bins[:-1], cum3d,label='3D, '+label[i])
        plt.ylim(0,1.1)
        plt.xlim(0,100)
        plt.legend(loc=0)

    return fig
   