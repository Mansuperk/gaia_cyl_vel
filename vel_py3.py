"""
VELOCITY VECTOR IN LSR SYSTEM
- VX towards galactic center
- VY towards galactic rotation direction
- VZ towards galactic north pole
VELOCITY IN CYLINDRICAL COORDINATES CENTERED IN GALACTIC CENTRE
-Vrho towards sun
-Vz towards galactic north pole
-Vphi in the direction of galactic rotation 

05/04/2023 - Manuel Ramos
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy as sc
import operator
from astropy.io import fits
from astropy.coordinates.representation import CartesianRepresentation, CylindricalRepresentation


#fits_table = fits.util.get_testdata_filepath('DR3_full.fits')
hdul = fits.open('DR3.fits')
data = hdul[1].data
nstars = data.size

k = 4.74047 #Proper motions constant

def vrpmllpmbb_to_vxvyvz(vr,pmll,pmbb,l,b,d,XYZ,degree):
    """                                                                                   
    NAME:                                                                                 
       vrpmllpmbb_to_vxvyvz                                                               
    PURPOSE:                                                                              
       Transform velocities in the spherical Galactic coordinate frame to the rectangular\
 Galactic coordinate frame                                                                
       Can take vector inputs                                                             
    INPUT:                                                                                
       vr - line-of-sight velocity (km/s)                                                 
       pmll - proper motion in the Galactic longitude (mu_l * cos(b))(mas/yr)             
       pmbb - proper motion in the Galactic lattitude (mas/yr)                            
       l - Galactic longitude                                                             
       b - Galactic lattitude                                                             
       d - distance (kpc)                                                                 
       XYZ - (bool) If True, then l,b,d is actually X,Y,Z (rectangular Galactic coordinates)                                                                                       
       degree - (bool) if True, l and b are in degrees                                    
    OUTPUT:                                                                               
       (vx,vy,vz) in (km/s,km/s,km/s)                                                     
    HISTORY:                                                                              
       2009-10-24 - Written - Bovy (NYU)                                                  
    """
    
    if XYZ:
        lbd = XYZ_to_lbd(l,b,d,degrees=True)
        if degree:
            l = np.deg2rad(lbd[0])
            b = np.deg2rad(lbd[1])
        else:
            l = lbd[0]
            d = lbd[1]
        d = lbd [2]
    else:
        if degree:
            l = np.deg2rad(l)
            b = np.deg2rad(b)
    
    R = np.zeros((3,3))
    R[0,0] = np.cos(l)*np.cos(b)
    R[1,0] = -np.sin(l)
    R[2,0] = -np.cos(l)*np.sin(b)
    R[0,1] = np.sin(l)*np.cos(b)
    R[1,1] = np.cos(l)
    R[2,1] = -np.sin(l)*np.sin(b)
    R[0,2] = np.sin(b)
    R[2,2] = np.cos(b)
    
    vxvyvz = np.dot(np.transpose(R),np.array([vr,d*pmll*k,d*pmbb*k]))
    return (vxvyvz[0],vxvyvz[1],vxvyvz[2])

def XYZ_to_lbd(X,Y,Z,degree = False):
    
    d = np.sqrt(X**2+Y**2+Z**2)
    b = np.asin(Z/d)
    cosl = X/d/np.cos(b)
    sinl = Y/d/np.cos(b)
    l = np.asin(sinl)
    
    if cosl < 0:
        l = np.pi - l
    elif sinl < 0:
        l = 2*np.pi + l
    if degree:
        return [l/np.pi*180, b/np.pi*180, d]
    else:
        return [l,b,d]
    
    
def get_epoch_angles(epoch=2000.0):
    """                                                                                                                        
    NAME:                                                                                                                      
       get_epoch_angles                                                                                                        
    PURPOSE:                                                                                                                   
       get the angles relevant for the transformation from ra, dec to l,b                                                      
       for the given epoch                                                                                                     
    INPUT:                                                                                                                     
       epoch - epoch of ra,dec (right now only 2000.0 and 1950.0 are supported                                                 
    OUTPUT:                                                                                                                    
       set of angles                                                                                                           
    HISTORY:                                                                                                                   
       2010-04-07 - Written - Bovy (NYU)                                                                                       
    """
    if epoch == 2000.0:
        theta= 122.932/180.*np.pi
        dec_ngp= 27.12825/180.*np.pi
        ra_ngp= 192.85948/180.*np.pi
    elif epoch == 1950.0:
        theta= 123./180.*np.pi
        dec_ngp= 27.4/180.*np.pi
        ra_ngp= 192.25/180.*np.pi
    elif epoch == 2015.0:
        theta = 0. ###mentira
        ra_ngp = 192.861625/180.*np.pi
        dec_ngp = 27.046927/180.*np.pi
    else:
        print("Only epochs 1950  2000 and 2015 are supported")
        print("Returning...")
        return -1
    return [theta,dec_ngp,ra_ngp]


def pmrapmdec_to_pmllpmbb_single(pmra,pmdec,ra,dec,b,degree=True,epoch=2000.0):
    """                                                                                                                        
    NAME:                                                                                                                      
       pmrapmdec_to_pmllpmbb_single                                                                                            
    PURPOSE:                                                                                                                   
       rotate proper motions in (ra,dec) into proper motions in (l,b) for                                                      
       scalar inputs                                                                                                           
    INPUT:                                                                                                                     
       pmra - proper motion in ra (multplied with cos(dec)) [mas/yr]                                                           
       pmdec - proper motion in dec [mas/yr]                                                                                   
       ra - right ascension                                                                                                    
       dec - declination                                                                                                       
       b - Galactic lattitude                                                                                                  
       degree - if True, ra and dec are given in degrees (default=False)                                                       
       epoch - epoch of ra,dec (right now only 2000.0 and 1950.0 are supported)                                                
    OUTPUT:                                                                                                                    
       (pmll,pmbb)                                                                                                             
    HISTORY:                                                                                                                   
       2010-04-07 - Written - Bovy (NYU)                                                                                       
    """

    theta,dec_ngp,ra_ngp= get_epoch_angles(epoch)
    if degree:
        sindec_ngp= np.sin(dec_ngp)
        cosdec_ngp= np.cos(dec_ngp)
        sindec= np.sin(np.deg2rad(dec))
        sinb= np.sin(np.deg2rad(b))
        cosdec= np.cos(np.deg2rad(dec))
        cosb= np.cos(np.deg2rad(b))
        sinrarangp= np.sin(np.deg2rad(ra)-ra_ngp)
    else:
        sindec_ngp= np.sin(dec_ngp)
        cosdec_ngp= np.cos(dec_ngp)
        sindec= np.sin(dec)
        sinb= np.sin(b)
        cosdec= np.cos(dec)
        cosb= np.cos(b)
        sinrarangp= np.sin(ra-ra_ngp)
    cosphi= (sindec_ngp-sindec*sinb)/cosdec/cosb
    sinphi= sinrarangp*cosdec_ngp/cosb
    out= np.dot(np.array([[cosphi,sinphi],[-sinphi,cosphi]]),np.array([pmra,pmdec]))
    return (out[0], out[1])

def movingaverage(values,window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def car2cyl(x,y,z): #Cartesian coordiantes to cylindrical coordinates
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    z = z
    return [rho,phi,z]

def vel_car2cyl(vx,vy,vz,phi): #Velocity in cartesian system to cylindrical coordinates
    vrho = vx*np.cos(phi)+vy*np.sin(phi)
    vphi = -vx*np.sin(phi)+vy*np.cos(phi)
    vz = vz
    return [vrho,vphi,vz]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#SOLAR MOTION (km/s) (Antoja et al. 2018)
U0 = 11.1
V0 = 12.24
W0 = 7.25
R_sun = 8340 #pc
V_R_sun = 240

stars = nstars

with open('V_LSR_1000.csv','w') as f:
    
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\n'.format('#','RA','DEC','PMRA','PMDEC','PLX','PLX_ERR','G_mag','BP_RP','grvs','temp_teff','Teff','logg','mh','RV','L','L_disp','B','DIST_1','DIST_500','DIST_1000','X','Y','Z','X_500','Y_500','Z_500','X_1000','Y_1000','Z_1000','X_gal', 'Y_gal', 'Z_gal','X_gal_500','Y_gal_500','Z_gal_500','X_gal_1000','Y_gal_1000','Z_gal_1000','rho_gal','phi_gal','rho_gal_500','phi_gal_500','rho_gal_1000','phi_gal_1000','VX_1','VY_1','VZ_1','VX_gal_1', 'VY_gal_1', 'VZ_gal_1','Vrho_1','Vphi_1','Vz_1','Vrho_gal_1', 'Vphi_gal_1','Vz_gal_1','VX_500','VY_500','VZ_500','VX_gal_500', 'VY_gal_500', 'VZ_gal_500','Vrho_500','Vphi_500','Vz_500','Vrho_gal_500', 'Vphi_gal_500', 'Vz_gal_500', 'VX_1000','VY_1000','VZ_1000', 'VX_gal_1000', 'VY_gal_1000', 'VZ_gal_1000','Vrho_1000','Vphi_1000','Vz_1000','Vrho_gal_1000', 'Vphi_gal_1000', 'Vz_gal_1000'))
#    f.write('{},{},{}'.format('VX_LSR_1000', 'VY_LSR_1000', 'VZ_LSR_1000'))
    
    for i in range(stars-1):
        RA = data['ra'][i]
        DEC = data['dec'][i]
        PMRA = data['pmra'][i]
        PMDEC = data['pmdec'][i]
        DIST_1 = data['dist_1'][i]
        DIST_500 = data['dist_500pc'][i]
        DIST_1000 = data['dist_1000pc'][i]
        L = data['l'][i]
        B = data['b'][i]
        RV = data['radial_velocity'][i]
        grvs = data['grvs_mag'][i]
        temp_teff = data['rv_template_teff'][i]
        PLX = data['parallax'][i]
        PLX_ERR = data['parallax_error'][i]
        G_mag = data['phot_g_mean_mag'][i]
        BP_RP = data['bp_rp'][i]
        Teff = data['teff_gspphot'][i]
        logg = data['logg_gspphot'][i]
        mh = data['mh_gspphot'][i]
        
        if L > 180: #Milky Way with the bulge at 180º
            L_disp = L - 180
        if L < 180:
            L_disp = L + 180
        
        if grvs >= 11 and temp_teff < 8500: #(Katz et al., 2022)
            RV = RV - 0.02755*grvs**2 + 0.55863*grvs - 2.81129
    
        PM_L,PM_B = pmrapmdec_to_pmllpmbb_single(PMRA, PMDEC, RA, DEC, B, True, 2000.0)
    
        Vl_1 = k*PM_L*DIST_1/1000.
        Vb_1 = k*PM_B*DIST_1/1000.
        Vl_500 = k*PM_L*DIST_1/1000.
        Vb_500 = k*PM_B*DIST_1/1000.
        Vl_1000 = k*PM_L*DIST_1/1000.
        Vb_1000 = k*PM_B*DIST_1/1000.
        
        RA_rad = np.deg2rad(RA)
        DEC_rad = np.deg2rad(DEC)
        L_rad = np.deg2rad(L)
        B_rad = np.deg2rad(B)
        
        """
        _1: distance calculated with 1/parallax
        _500: distance estimated with prior = 500 pc
        _1000: distance estimated with prior = 1000 pc
        """
        vxvyvz_1 = vrpmllpmbb_to_vxvyvz(RV,PM_L,PM_B,L,B,DIST_1/1000.,False,True) 
        
        vxvyvz_500 = vrpmllpmbb_to_vxvyvz(RV,PM_L,PM_B,L,B,DIST_500/1000.,False,True)
        
        vxvyvz_1000 = vrpmllpmbb_to_vxvyvz(RV,PM_L,PM_B,L,B,DIST_1000/1000.,False,True)
    
        VX_1 = vxvyvz_1[0]
        VY_1 = vxvyvz_1[1]
        VZ_1 = vxvyvz_1[2]
        
        VX_500 = vxvyvz_500[0]
        VY_500 = vxvyvz_500[1]
        VZ_500 = vxvyvz_500[2]
        
        VX_1000 = vxvyvz_1000[0]
        VY_1000 = vxvyvz_1000[1]
        VZ_1000 = vxvyvz_1000[2]
       
        VX_LSR_1 = U0 + VX_1
        VY_LSR_1 = V0 + VY_1
        VZ_LSR_1 = W0 + VZ_1
        
        VX_LSR_500 = U0 + VX_500
        VY_LSR_500 = V0 + VY_500
        VZ_LSR_500 = W0 + VZ_500
        
        VX_LSR_1000 = U0 + VX_1000
        VY_LSR_1000 = V0 + VY_1000
        VZ_LSR_1000 = W0 + VZ_1000
          
        X, Y, Z = data['XYZ_gal'][i]
        X_500, Y_500, Z_500 = data['XYZ_gal_500'][i]
        X_1000, Y_1000, Z_1000 = data['XYZ_gal_1000'][i]
        
        rho, phi, z = car2cyl(X, Y, Z)
        rho_500, phi_500, z_500 = car2cyl(X_500, Y_500, Z_500)
        rho_1000, phi_1000, z_1000 = car2cyl(X_1000, Y_1000, Z_1000)
        
        X_gal = X + R_sun
        Y_gal = Y
        Z_gal = Z
       
        X_gal_500 = X_500 + R_sun
        Y_gal_500 = Y_500
        Z_gal_500 = Z_500
        
        X_gal_1000 = X_1000 + R_sun
        Y_gal_1000 = Y_1000
        Z_gal_1000 = Z_1000
        
        rho_gal, phi_gal, z_gal = car2cyl(X_gal, Y_gal, Z_gal)
        rho_gal_500, phi_gal_500, z_gal_500 = car2cyl(X_gal_1000, Y_gal_1000, Z_gal_1000)
        rho_gal_1000, phi_gal_1000, z_gal_1000 = car2cyl(X_gal_1000, Y_gal_1000, Z_gal_1000)
        
        VX_gal_1 = -VX_LSR_1
        VY_gal_1 = VY_LSR_1 + V_R_sun
        VZ_gal_1 = VZ_LSR_1
        
        VX_gal_500 = -VX_LSR_500
        VY_gal_500 = VY_LSR_500 + V_R_sun
        VZ_gal_500 = VZ_LSR_500
        
        VX_gal_1000 = -VX_LSR_1000
        VY_gal_1000 = VY_LSR_1000 + V_R_sun
        VZ_gal_1000 = VZ_LSR_1000
        
        Vrho_1, Vphi_1, Vz_1 = vel_car2cyl(VX_1, VY_1, VZ_1, phi)
        Vrho_LSR_1, Vphi_LSR_1, Vz_LSR_1 = vel_car2cyl(VX_LSR_1, VY_LSR_1, VZ_LSR_1, phi)
        Vrho_gal_1, Vphi_gal_1, Vz_gal_1 = vel_car2cyl(VX_gal_1, VY_gal_1, VZ_gal_1, phi_gal)
        
        Vrho_500, Vphi_500, Vz_500 = vel_car2cyl(VX_500, VY_500, VZ_500, phi_500)
        Vrho_LSR_500, Vphi_LSR_500, Vz_LSR_500 = vel_car2cyl(VX_LSR_500, VY_LSR_500, VZ_LSR_500, phi_500)
        Vrho_gal_500, Vphi_gal_500, Vz_gal_500= vel_car2cyl(VX_gal_500, VY_gal_500, VZ_gal_500, phi_gal_500)
        
        Vrho_1000, Vphi_1000, Vz_1000 = vel_car2cyl(VX_1000, VY_1000, VZ_1000, phi_1000)
        Vrho_LSR_1000, Vphi_LSR_1000, Vz_LSR_1000 = vel_car2cyl(VX_LSR_1000, VY_LSR_1000, VZ_LSR_1000, phi_1000)
        Vrho_gal_1000, Vphi_gal_1000, Vz_gal_1000 = vel_car2cyl(VX_gal_1000, VY_gal_1000, VZ_gal_1000, phi_gal_1000)
        
#        VX_cyl, VY_cyl, VZ_cyl = CartesianRepresentation(VX, VY, VZ).represent_as(CylindricalRepresentation)
#        VX_cyl_LSR, VY_cyl_LSR, VZ_cyl_LSR = CartesianRepresentation(VX_LSR, VY_LSR, VZ_LSR).represent_as(CylindricalRepresentation)
        
        vxvyvz_top = data['UVW_gal'][i]
        VX_top = vxvyvz_top[0]
        VY_top = vxvyvz_top[1]
        VZ_top = vxvyvz_top[2]
        
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\n'.format(i+1,RA,DEC,PMRA,PMDEC,PLX,PLX_ERR,G_mag,BP_RP,grvs,temp_teff,Teff,logg,mh,RV,L,L_disp,B,DIST_1,DIST_500,DIST_1000,X,Y,Z,X_500,Y_500,Z_500,X_1000,Y_1000,Z_1000,X_gal,Y_gal,Z_gal,X_gal_500,Y_gal_500,Z_gal_500,X_gal_1000,Y_gal_1000,Z_gal_1000,rho_gal,phi_gal,rho_gal_500,phi_gal_500,rho_gal_1000,phi_gal_1000,VX_1,VY_1,VZ_1,VX_gal_1,VY_gal_1,VZ_gal_1,Vrho_1,Vphi_1,Vz_1,Vrho_gal_1,Vphi_gal_1,Vz_gal_1,VX_500,VY_500,VZ_500,VX_gal_500,VY_gal_500,VZ_gal_500,Vrho_500,Vphi_500,Vz_500,Vrho_gal_500,Vphi_gal_500,Vz_gal_500,VX_1000,VY_1000,VZ_1000,VX_gal_1000,VY_gal_1000,VZ_gal_1000,Vrho_1000,Vphi_1000,Vz_1000,Vrho_gal_1000,Vphi_gal_1000,Vz_gal_1000))
#        f.write('{}, {}, {}'.format(VX_LSR_1000, VY_LSR_1000, VZ_LSR_1000))       
        if i % 100000 == 0:
            pc = (i/stars)*100
            pc = int(pc)
            print('{} %'.format(pc))