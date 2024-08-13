#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-03-02

@author: alexaputnam
"""
import os
import time
import pyproj
import geopandas
import numpy as np
import earthaccess
import datetime as dt
import rasterio as rio
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.interpolate import NearestNDInterpolator

#########################################################################################################################
#########################################################################################################################
####### GLOBAL DEFINITIONS to be adjusted by user
#########################################################################################################################
#########################################################################################################################
# Define a path where Earthdata can temporarily download data to. Once downloaded, the function pulls out the 
#   data of interest and then deletes the file within the pth_temp.
pth_temp = '/temporary/directory/to/download/pull/delete/product/data/swot_earthdata_temp/'

AUTHOR = "Alexa Putnam"
INSTITUTION = "University of Colorado Boulder"
SAVETAG = 'Putnam'

#########################################################################################################################
#########################################################################################################################
####### Example
#########################################################################################################################
#########################################################################################################################
'''
# Example for Ocean
# Input
LOC = 'nuuk'
LLMM = [64, 65, -53, -49]
yrS,mnS,dyS,hrS = 2024,6,21,0
yrE,mnE,dyE,hrE = 2024,7,18,23
PTHSAV = /aputnam/example/
PASS = '257'
SCENE = []
SN = 'SWOT_L2_LR_SSH_2.0'
GN = []

# Run
earth_data_route_ocean(REG,LLMM,yrS,mnS,dyS,0,yrE,mnE,dyE,23,PTHSAV,PASS=PASS)

# Output
/aputnam/example/SWOT_L2_LR_SSH_nuuk_Putnam_017_257.nc
or
PTHSAV/MISSION_PRODUCT_LOC_SAVETAG_CYCLE_PASS.nc


'''

#########################################################################################################################
#########################################################################################################################
####### MAIN FUNCTION
#########################################################################################################################
#########################################################################################################################


def earth_data_route(LOC,LLMM,yrS,mnS,dyS,hrS,yrE,mnE,dyE,hrE,PTHSAV,PASS=[],SCENE=[],SN='SWOT_L2_HR_Raster_2.0',GN=[]):
    # https://swot.jpl.nasa.gov/mission/swath-visualizer/#:~:text=Pass%20%2D%20SWOT%20retraces%20the%20same,starts%20near%20the%20South%20Pole.
    #https://podaac.github.io/tutorials/quarto_text/SWOT.html
    '''
    LOC: (string) to indicate the location of the dataset (i.e. LOC == 'Mekong_River'). This is solely for saving the file.
    LLMM: (array) defines the search area as a rectangle with points of min/max lat & lon. LLMM =[minimum lat, maximum lat, minimum lon, maximum lon] (i.e. LLMM = [10,12,104,106])
    yrS,mnS,dyS,hrS: (integers) starting year, month, day and hour for Earthdata search, respectively (i.e. yrS,mnS,dyS,hrS=2023,1,30,0 for January 30th, 2023 at 12 AM)
    yrE,mnE,dyE,hrE: (integers) ending year, month, day and hour for Earthdata search, respectively (i.e. yrE,mnE,dyE,hrE=2024,7,15,20 for July 17th, 2024 at 8 PM)
    PTHSAV: (string) path to the directory where the output file will be saved
    PASS: (string) 3-digit pass number if available. This helps to speed up the process. If not defined, the function will search all passes. (i.e. PASS = '257' for Nuuk estuary)
    SCENE: (string) scene if available. 
        PIXC: 3-digit scene number (i.e. PASS='089' and SCENE='148L' for specific location over Amazon)
        Raster: 3-digit scene number (i.e. PASS='299' and SCENE='089F' for specific location over Tonle Sap, Cambodia)
        RiverSP:scene defined as continent code (i.e. 'NA' for North America)
        L2_LR_SSH: []
    SN: (string) SWOT short name of main product. Default SN='SWOT_L2_HR_Raster_2.0' for the Raster product.
        PIXC: SN='SWOT_L2_HR_PIXC_2.0'
        Raster: SN='SWOT_L2_HR_Raster_2.0'
        RiverSP: SN='SWOT_L2_HR_RiverSP_2.0'
        L2_LR_SSH: SN ='SWOT_L2_LR_SSH_2.0'
    GN: (string) SWOT granule name if availble. 
        PIXC: GN = []
        Raster: GN = '100m' or '250m' depending on resolution of interest
        RiverSP: GN = 'Node' or 'Reach' depending on product of interest
        L2_LR_SSH: []
    '''
    auth = earthaccess.login() #persist=True
    #yrS,mnS,dyS,hrS,yrE,mnE,dyE,hrE = is2_sec_2_ymd(utcIS85,dt_buff_days)
    start_data = str(yrS)+'-'+'{0:0=2d}'.format(mnS)+'-'+'{0:0=2d}'.format(dyS)+' '+'{0:0=2d}'.format(hrS)+':00:00' ## start_data = '2023-01-01 00:00:00'
    end_data = str(yrE)+'-'+'{0:0=2d}'.format(mnE)+'-'+'{0:0=2d}'.format(dyE)+' '+'{0:0=2d}'.format(hrE)+':59:59' ## end_data = '2023-08-31 23:59:59'
    print('start: '+start_data)
    print('end: '+end_data)
    if SN not in ['SWOT_L2_HR_PIXC_2.0','SWOT_L2_HR_Raster_2.0','SWOT_L2_HR_RiverSP_2.0','SWOT_L2_LR_SSH_2.0']:
        print('short_name not in: [SWOT_L2_HR_PIXC_2.0, SWOT_L2_HR_Raster_2.0, SWOT_L2_HR_RiverSP_2.0, SWOT_L2_LR_SSH_2.0]')
        raise('Functions may not be made for this project. Please double check')
    if SN=='SWOT_L2_LR_SSH_2.0':
        earth_data_route_ocean(LOC,LLMM,yrS,mnS,dyS,hrS,yrE,mnE,dyE,hrE,PTHSAV,PASS=PASS)
    else:
        if np.size(GN)!=0:
            GN = '*_'+GN+'_*'
        else:
            GN = '*'
        if np.size(PASS)!=0: # PASS='510'
            GN = GN+str(PASS)+'_*' # GNus = '*Unsmoothed*_560_020_*'
        if np.size(SCENE)!=0: # PASS='510'
            GN = GN+str(SCENE)+'_*' # GNus = '*Unsmoothed*_560_020_*'

        # Raster data
        print('short_name: '+SN)
        print('granule_name: '+GN)
        ###
        results_ex = earthaccess.search_data(short_name = SN,temporal = (start_data, end_data),granule_name = GN) # here we filter by Reach files (not node), pass #13 and continent code=NA
        N_ex = np.shape(results_ex)[0]
        cyc_ex=[]
        pass_ex=[]
        for ii in np.arange(N_ex):
            fnii_ex = results_ex[ii]['meta']['native-id']
            if SN=='SWOT_L2_HR_Raster_2.0':
                print('Raster: '+fnii_ex[38:45])
                cp_ex = fnii_ex[38:45]
            elif SN=='SWOT_L2_HR_RiverSP_2.0':
                if 'Node' in GN:
                    print('Node: '+fnii_ex[24:31])
                    cp_ex = fnii_ex[24:31]#fnii_ex[38:45]
                elif 'Reach' in GN:
                    print('Reach: '+fnii_ex[25:32])
                    cp_ex = fnii_ex[25:32]
            elif SN=='SWOT_L2_HR_PIXC_2.0':
                print('PIXC: '+fnii_ex[20:27])
                cp_ex = fnii_ex[20:27]
            cyc_ex.append(int(cp_ex[:3]))
            pass_ex.append(int(cp_ex[4:]))  
        cyc_ex = np.asarray(cyc_ex)
        pass_ex = np.asarray(pass_ex)
        # Download
        print('Number of results = '+str(N_ex))
        for ii in np.arange(N_ex):
                iex = np.where((cyc_ex[ii]==cyc_ex))[0] #&(pass_us[ii]==pass_ex))[0]
                if np.size(iex)!=0:
                    earthaccess.download(results_ex[iex[0]], pth_temp)
                    # Filter and save as netCDF
                    if 'SWOT_L2_HR_RiverSP' in results_ex[iex[0]]['meta']['native-id']:
                        fnii_ex = results_ex[iex[0]]['meta']['native-id']+'.zip'
                        FILESV = pthSV+fnii_ex[:-4]+'_'+LOC+'_'+SAVETAG+'.nc'
                        if 'Node' in GN:
                            print('File to pull_and_save_node: '+fnii_ex)
                            pull_and_save_node(fnii_ex,LLMM,FILESV=FILESV)
                        elif 'Reach' in GN:
                            print('File to pull_and_save_reach: '+fnii_ex) 
                            pull_and_save_reach(fnii_ex,LLMM,FILESV=FILESV)
                    elif 'SWOT_L2_HR_Raster' in results_ex[iex[0]]['meta']['native-id']:
                        fnii_ex = results_ex[iex[0]]['meta']['native-id']+'.nc'
                        print('File to pull_and_save_raster: '+fnii_ex)
                        FILESV = pthSV+fnii_ex[:-3]+'_'+LOC+'_'+SAVETAG+'.nc'
                        pull_and_save_raster(fnii_ex,LLMM,FILESV=FILESV)
                    elif 'SWOT_L2_HR_PIXC' in results_ex[iex[0]]['meta']['native-id']:
                        fnii_ex = results_ex[iex[0]]['meta']['native-id']+'.nc'
                        print('File to pull_and_save_pixc: '+fnii_ex)
                        DT=dt.datetime.today().strftime('%Y%m%d%H%M')
                        FILESV = pthSV+fnii_ex[:-3]+'_'+LOC+'_'+SAVETAG+'_'+DT+'.nc'
                        pull_and_save_pixc(fnii_ex,LLMM,FILESV=FILESV)
                        # erase downloaded files
                    os.system("rm "+pth_temp+fnii_ex)


####### SUB-MAIN FUNCTION for Ocean only
def earth_data_route_ocean(LOC,LLMM,yrS,mnS,dyS,hrS,yrE,mnE,dyE,hrE,PTHSAV,PASS=[]):
    # https://swot.jpl.nasa.gov/mission/swath-visualizer/#:~:text=Pass%20%2D%20SWOT%20retraces%20the%20same,starts%20near%20the%20South%20Pole.
    #https://podaac.github.io/tutorials/quarto_text/SWOT.html
    '''
    LOC: (string) to indicate the location of the dataset (i.e. LOC == 'Mekong_River'). This is solely for saving the file.
    LLMM: (array) defines the search area as a rectangle with points of min/max lat & lon. LLMM =[minimum lat, maximum lat, minimum lon, maximum lon] (i.e. LLMM = [10,12,104,106])
    yrS,mnS,dyS,hrS: (integers) starting year, month, day and hour for Earthdata search, respectively (i.e. yrS,mnS,dyS,hrS=2023,1,30,0 for January 30th, 2023 at 12 AM)
    yrE,mnE,dyE,hrE: (integers) ending year, month, day and hour for Earthdata search, respectively (i.e. yrE,mnE,dyE,hrE=2024,7,15,20 for July 17th, 2024 at 8 PM)
    PTHSAV: (string) path to the directory where the output file will be saved
    PASS: (string) pass number if available. This helps to speed up the process. If not defined, the function will search all passes. (i.e. PASS = '257' for Nuuk estuary)
    '''
    auth = earthaccess.login() #persist=True
    start_data = str(yrS)+'-'+'{0:0=2d}'.format(mnS)+'-'+'{0:0=2d}'.format(dyS)+' '+'{0:0=2d}'.format(hrS)+':00:00' ## start_data = '2023-01-01 00:00:00'
    end_data = str(yrE)+'-'+'{0:0=2d}'.format(mnE)+'-'+'{0:0=2d}'.format(dyE)+' '+'{0:0=2d}'.format(hrE)+':59:59' ## end_data = '2023-08-31 23:59:59'
    print('start: '+start_data)
    print('end: '+end_data)
    SN='SWOT_L2_LR_SSH_2.0' #1.1
    if np.size(PASS)!=0: # PASS='510'
        GNus = '*Unsmoothed*_'+PASS+'_*' # GNus = '*Unsmoothed*_560_020_*'
        GNex='*Expert*_'+PASS+'_*' # GNex='*Expert*_560_020_*'
    else:
        GNus = '*Unsmoothed*'
        GNex='*Expert*'
    # Expert data
    print('short_name: '+SN)
    print('granule_name 1: '+GNus)
    print('granule_name 2: '+GNex)
    results_ex = earthaccess.search_data(short_name = SN, 
                                temporal = (start_data, end_data),
                                granule_name = GNex) # here we filter by Reach files (not node), pass #13 and continent code=NA
    N_ex = np.shape(results_ex)[0]
    cyc_ex=[]
    pass_ex=[]
    for ii in np.arange(N_ex):
        fnii_ex = results_ex[ii]['meta']['native-id']
        cp_ex = fnii_ex[22:29]
        cyc_ex.append(int(cp_ex[:3]))
        pass_ex.append(int(cp_ex[4:]))  
    cyc_ex = np.asarray(cyc_ex)
    pass_ex = np.asarray(pass_ex)
    # Unsmoothed data
    results_us = earthaccess.search_data(short_name = SN, 
                                temporal = (start_data, end_data),
                                granule_name = GNus) # here we filter by Reach files (not node), pass #13 and continent code=NA
    N_us = np.shape(results_us)[0]
    cyc_us=[]
    pass_us=[]
    for ii in np.arange(N_us):
        fnii_us = results_us[ii]['meta']['native-id']
        cp_us = fnii_us[26:33]
        cyc_us.append(int(cp_us[:3]))
        pass_us.append(int(cp_us[4:]))
    cyc_us = np.asarray(cyc_us)
    pass_us = np.asarray(pass_us)

    # Download
    print('Number of results = '+str(N_us))
    for ii in np.arange(N_us):
            iex = np.where((cyc_us[ii]==cyc_ex)&(pass_us[ii]==pass_ex))[0]
            if np.size(iex)!=0:
                fnii_us = results_us[ii]['meta']['native-id']+'.nc'
                fnii_ex = results_ex[iex[0]]['meta']['native-id']+'.nc'
                print(fnii_ex)#SWOT_L2_LR_SSH_Expert_560_020_20230623T004717_20230623T013423_PIB0_01.nc
                print(fnii_us)#SWOT_L2_LR_SSH_Unsmoothed_560_020_20230623T004708_20230623T013813_PIB0_01.nc
                earthaccess.download(results_us[ii], pth_temp)
                earthaccess.download(results_ex[iex[0]], pth_temp)
                # Filter and save as netCDF
                FILESV = PTHSAV+fnii_us[:15]+LOC+'_'+SAVETAG+'_'+fnii_us[26:33]+'.nc'
                pull_expert_and_unsmoothed_swot(fnii_ex,fnii_us,LLMM,FILESV=FILESV)
                # erase downloaded files
                os.system("rm "+pth_temp+fnii_us)
                os.system("rm "+pth_temp+fnii_ex)
    
    #folder = Path(pth_temp)


#########################################################################################################################
#########################################################################################################################
####### Random FUNCTIONS
#########################################################################################################################
#########################################################################################################################

def lla2ecef(lat_deg,lon_deg):
    #x,y,z = lla2ecef(lat_deg,lon_deg)
    # WGS84 ellipsoid constants:
    alt = 0.
    a = 6378137. #height above WGS84 ellipsoid (m)
    e = 8.1819190842622e-2
    d2r = np.pi/180.
    N = a/np.sqrt(1.-(e**2)*(np.sin(lat_deg*d2r)**2))
    x = (N+alt)*np.cos(lat_deg*d2r)*np.cos(lon_deg*d2r)
    y = (N+alt)*np.cos(lat_deg*d2r)*np.sin(lon_deg*d2r)
    z = ((1.-e**2)*N+alt)*np.sin(lat_deg*d2r)
    return x,y,z

def dist_func(xA,yA,zA,xD,yD,zD):
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist

def geo2dist(lat,lon,lat0,lon0):
    # Calculate the distance between lat,lon and lat0,lon0
    # lat,lon = arrays of latitude and longitude [degrees]
    # lat0,lon0 = a single coordinate of latitude and longitude [degrees] to be used as a reference for which to calculate distance
    # output: distance (dist) in meters
    lon360 = lon180_to_lon360(lon)
    x,y,z = lla2ecef(lat,lon360)
    lon360ref = lon180_to_lon360(lon0)
    xr,yr,zr = lla2ecef(lat0,lon360ref)
    dist = dist_func(xr,yr,zr,x,y,z)
    idx = np.where((lat-lat0)<0)[0]
    if np.size(idx)>1:
        dist[idx]=dist[idx]*-1.0
    elif np.size(idx)==1:
        dist=dist*-1.0
    return dist

def lon360_to_lon180(lon_old):
    # Convert longitude from 0:360 to -180:180
    igt = np.where(lon_old>180)[0] #!!! used to be (prior to may 5 2021) lon_old>=180
    if np.size(igt)!=0:
        lon_new = np.mod((lon_old+180.),360.)-180.
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def lon180_to_lon360(lon_old):
    # Convert longitude from -180:180 to 0:360
    igt = np.where(lon_old<0)[0]
    if np.size(igt)!=0:
        lon_new = np.mod(lon_old,360.)
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def utc2yrfrac(utc,YEAR=1985):
    # Convert UTC time to year fraction given seconds (utc) since some year (YEAR)
    # Default: YEAR = 1985 (YEAR used by ICESat-2. SWOT uses 2000.)
    N = np.shape(utc)[0]
    utc_yr = utc/(60.*60.*24.*365.25)
    yrfrac = utc_yr+YEAR
    return yrfrac

def alpha2num(LIST):
    # River IDs
    Nl = np.shape(LIST)[0]
    chr1 = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    chr2 = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    nmb = ['901','902','903','904','905','906','907','908','910','911','912','913','914','915','916','917','918','920','921','922','923','924','925','926','927','928']
    ARR=[]
    for ii in np.arange(Nl):
        WRD = LIST[ii]
        Nw = len(WRD)
        RIV=''
        for jj in np.arange(Nw):
            LETT = WRD[jj]
            if LETT in chr1:
                ich = np.where(chr1==LETT)[0][0]
                RIV=RIV+nmb[ich]
            elif LETT in chr2:
                ich = np.where(chr2==LETT)[0][0]
                RIV=RIV+nmb[ich]
        ARR.append(int(RIV))
    ARR = np.asarray(ARR)
    return ARR

def is2_sec_2_ymd(utcIS_85,dt_buff_days):
    gps2utc = (dt.datetime(1985, 1, 1,0,0,0)-dt.datetime(1980, 1, 1,0,0,0)).total_seconds()
    utc802py = (dt.datetime(1980, 1, 1,0,0,0)-dt.datetime(1970, 1, 1,0,0,0)).total_seconds()

    utcIS = utcIS_85+gps2utc+18
    tstmp_start=dt.datetime.fromtimestamp((utcIS)+utc802py-(dt_buff_days*24*60*60),dt.timezone.utc)
    tstmp_end=dt.datetime.fromtimestamp((utcIS)+utc802py+(dt_buff_days*24*60*60),dt.timezone.utc)
    yrS=tstmp_start.year
    mnS=tstmp_start.month
    dyS=tstmp_start.day
    hrS=tstmp_start.hour

    yrE=tstmp_end.year
    mnE=tstmp_end.month
    dyE=tstmp_end.day
    hrE=tstmp_end.hour
    return yrS,mnS,dyS,hrS,yrE,mnE,dyE,hrE

def interpolate_data(data_old,lat_old,lon_old,lat_new,lon_new):
    # Interpolate data (data_old) given its corrdinates (lat_old,lon_old) to new along-track points
    #   defined by a new set of coordinates (lat_new,lon_new)
    Nr,Nc=np.shape(lat_new)
    data_new = np.empty(np.shape(lat_new))*np.nan
    fdata_old,flat_old,flon_old = data_old.flatten('F'),lat_old.flatten('F'),lon_old.flatten('F')
    buff = 0.1
    for ii in np.arange(Nc):
        lonii = lon_new[:,ii]
        latii = lat_new[:,ii]
        inn = np.where((np.abs(fdata_old)<1000)&(flat_old>=np.nanmin(latii)-buff)&(flat_old<=np.nanmax(latii)+buff)&(flon_old>=np.nanmin(lonii)-buff)&(flon_old<=np.nanmax(lonii)+buff))[0]
        if np.size(inn)>0:
            f = NearestNDInterpolator(np.vstack((flon_old[inn], flat_old[inn])).T, fdata_old[inn])
            inn2 = np.where((~np.isnan(lonii))&(~np.isnan(latii)))[0]
            data_new[inn2,ii] = f(lonii[inn2], latii[inn2])          
    return data_new



#########################################################################################################################
#########################################################################################################################
####### HYDROLOGY FUNCTIONS
#########################################################################################################################
#########################################################################################################################

#########################################################################################################################
#########################################################################################################################
############## PIXC PRODUCT FUNCTIONS ############## 
def pixc_var(dsv_ex_pix,LLMM):
    # Quality flags
    """
    flag_meanings: no_coherent_gain power_close_to_noise_floor detected_water_but_no_prior_water detected_water_but_bright_land water_false_detection_rate_suspect coherent_power_suspect tvp_suspect sc_event_suspect small_karin_gap in_air_pixel_degraded specular_ringing_degraded coherent_power_bad tvp_bad sc_event_bad large_karin_gap
    flag_masks: [1, 2, 4, 8, 16, 2048, 8192, 16384, 32768, 262144, 524288, 134217728, 536870912,  1073741824,  2147483648]
    """
    class_qual = dsv_ex_pix['classification_qual'][:]
    coh_power = dsv_ex_pix['coherent_power'][:]
    geolocation_qual = dsv_ex_pix['geolocation_qual'][:]
    interferogram_qual= dsv_ex_pix['interferogram_qual'][:]
    sig0_qual = dsv_ex_pix['sig0_qual'][:]#0,1,2,3 = good, suspect, degraded and bad measurements
    pixc_line_qual = dsv_ex_pix['pixc_line_qual'][:]
    false_detection_rate = dsv_ex_pix['false_detection_rate'][:] # Probability of falsely detecting water when there is none.
    missed_detection_rate = dsv_ex_pix['missed_detection_rate'][:] # Probability of falsely detecting no water when there is water.
    prior_water_prob = dsv_ex_pix['prior_water_prob'][:] # Prior probability of water occurring.
    bright_land_flag = dsv_ex_pix['bright_land_flag'][:] # not_bright_land bright_land bright_land_or_water
    layover_impact = dsv_ex_pix['layover_impact'][:] #[m] Estimate of the height error caused by layover
    eff_num_rare_looks = dsv_ex_pix['eff_num_rare_looks'][:] # Effective number of independent looks taken to form the rare interferogram.
    # Geodetic longitude and latitude coordinates giving the 
    ## horizontal location of the center of the observed pixel. 
    ## The latitude is a geodetic latitude with respect to the 
    ## reference ellipsoid, which is defined by the semi_major_axis 
    ## and inverse_flattening attributes of the crs variable
    lon_all = dsv_ex_pix['longitude'][:]
    lat_all = dsv_ex_pix['latitude'][:]
    ikp = np.where((lat_all>=LLMM[0])&(lat_all<=LLMM[1])&(lon_all>=LLMM[2])&(lon_all<=LLMM[3]))[0]
    print('Size ikp: '+str(np.size(ikp)))
    if np.size(ikp)>0:
        NC = {}
        # Variables
        NC['lat']=lat_all[ikp]
        NC['lon']=lon_all[ikp]
        """
        MODIS/GlobCover
        flag_meanings: open_ocean land continental_water aquatic_vegetation continental_ice_snow floating_ice salted_basin
        flag_values: [0 1 2 3 4 5 6]
        """
        NC['surf_anc_qual'] = dsv_ex_pix['ancillary_surface_classification_flag'][:][ikp]
        """
        flag_meanings: land land_near_water water_near_land open_water dark_water low_coh_water_near_land open_low_coh_water
        flag_values: [1 2 3 4 5 6 7]
        """
        NC['surf_qual'] = dsv_ex_pix['classification'][:][ikp]
        NC['cross_track'] = dsv_ex_pix['cross_track'][:][ikp] # Approximate cross-track location of the pixel.
        NC['pixel_area'] = dsv_ex_pix['pixel_area'][:][ikp] # [m^2] pixel area
        NC['inc'] = dsv_ex_pix['inc'][:][ikp] #[deg] Incidence angle.
        NC['t_utc'] = dsv_ex_pix['illumination_time'][:][ikp] # time of illumination of each pixel (UTC). seconds since 2000-01-01 00:00:00.000. tai_utc_difference: 37.0
        NC['t_tai'] = dsv_ex_pix['illumination_time_tai'][:][ikp] # time of illumination of each pixel (TAI). seconds since 2000-01-01 00:00:00.000
        NC['sig0'] = dsv_ex_pix['sig0'][:][ikp] #Normalized radar cross section (sigma0) in real, linear units (not decibels). The value may be negative due to noise subtraction.
        NC['sig0_uncert'] = dsv_ex_pix['sig0_uncert'][:][ikp] # 1-sigma uncertainty in the sig0 measurement.  The value is given as an additive (not multiplicative) linear term (not a term in decibels).
        NC['height_cor_xover'] = dsv_ex_pix['height_cor_xover'][:][ikp] #Height correction from KaRIn crossover calibration. (already corrected)
        NC['model_dry_tropo_cor'] = dsv_ex_pix['model_dry_tropo_cor'][:][ikp] #ECMWF (already corrected)
        NC['model_wet_tropo_cor'] = dsv_ex_pix['model_wet_tropo_cor'][:][ikp] #ECMWF (already corrected)
        NC['iono_cor_gim_ka'] = dsv_ex_pix['iono_cor_gim_ka'][:][ikp] #GIM (already corrected)
        NC['geoid'] = dsv_ex_pix['geoid'][:][ikp] # EGM2008 (Pavlis et al., 2012) (not applied)
        NC['solid_earth_tide'] = dsv_ex_pix['solid_earth_tide'][:][ikp] #Cartwright and Taylor (1971) and Cartwright and Edden (1973)(not applied)
        NC['load_tide_fes'] = dsv_ex_pix['load_tide_fes'][:][ikp] #FES2014b (Carrere et al., 2016)  (not applied)
        NC['load_tide_got'] = dsv_ex_pix['load_tide_got'][:][ikp] #GOT4.10c (Ray, 2013)   (not applied)
        NC['pole_tide'] = dsv_ex_pix['pole_tide'][:][ikp] #Wahr (1985) and Desai et al. (2015) (not applied)
        NC['water_frac'] = dsv_ex_pix['water_frac'][:][ikp] # Noisy estimate of the fraction of the pixel that is water.
        NC['water_frac_unc'] = dsv_ex_pix['water_frac_uncert'][:][ikp] # Uncertainty estimate of the water fraction estimate (width of noisy water frac estimate distribution).
        NC['wse'] = dsv_ex_pix['height'][:][ikp]-NC['solid_earth_tide']-NC['load_tide_fes']-NC['pole_tide']-NC['geoid'] # Height of the pixel above the reference ellipsoid. [m]
    else:
        NC=0
    return NC

def pull_and_save_pixc(fn_ex,LLMM,FILESV=[]):
    FN = os.listdir(pth_temp)
    if fn_ex in FN:
        if 'SWOT_L2_HR_PIXC' in fn_ex:
            print('Open file')
            dsv_ex= Dataset(pth_temp+fn_ex)['pixel_cloud']
            print(dsv_ex.variables.keys())
            NC = pixc_var(dsv_ex,LLMM)
            if NC!=0:
                print('Saving '+FILESV)
                print('Data size: '+str(np.size(NC['geoid'])))
                root_grp = Dataset(FILESV, 'w', format='NETCDF4')
                root_grp.description = "This file contains SWOT_L2_HR_PIXC_ ... data filtered for LLMM = "+str(LLMM)
                root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
                #Nx,Ny=np.size(NC['x']),np.size(NC['y'])
                Nx = np.size(NC['lat'])
                root_grp.createDimension('X', Nx)
                #root_grp.createDimension('Y', Ny)
                var = NC.keys()
                for vv in var:
                    if vv not in ['x','y']:
                        print(vv)
                        print('size :'+str(np.size(NC[vv])))
                        xi = root_grp.createVariable(vv, 'f8', ('X'))
                        xi[:] = NC[vv]
                root_grp.close()


#########################################################################################################################
#########################################################################################################################
############## RASTER PRODUCT FUNCTIONS ############## 
def raster_var(dsv_ex,LLMM):
    x = dsv_ex['x'][:] # easting UTM projection
    y = dsv_ex['y'][:] # northing UTM projection
    mx,my = np.meshgrid(x,y)
    # Spatial Reference System EPSG 3857 (WGS84 Web Mercator)
    # Geographic coordinate system (EPSG 4326)
    inProj = pyproj.Proj(init='epsg:3857')
    outProj = pyproj.Proj(init='epsg:4326')
    mx2, my2 = pyproj.transform(inProj, outProj, mx, my)  
    transformer = Transformer.from_crs(3857, 4326, always_xy=True)
    mxT, myT = transformer.transform(mx, my)
    # Quality flags
    wse_qual = dsv_ex['wse_qual'][:] #0,1,2,3 = good, suspect, degraded and bad measurements
    water_area_qual = dsv_ex['water_area_qual'][:] #0,1,2,3 = good, suspect, degraded and bad measurements
    sig0_qual = dsv_ex['sig0_qual'][:]#0,1,2,3 = good, suspect, degraded and bad measurements
    # Number of samples from the L2_HR_PIXC product which contribute to the WSE of a given raster water pixel
    n_wse_pix = dsv_ex['n_wse_pix'][:]
    # Number of samples from the L2_HR_PIXC product which contribute to the water surface area and water fraction of a given raster water pixel
    n_water_area_pix = dsv_ex['n_water_area_pix'][:]
    # Number of samples from the L2_HR_PIXC product which contribute to the sigma0 of a given raster water pixel
    n_sig0_pix = dsv_ex['n_sig0_pix'][:]
    # Number of samples from the L2_HR_PIXC product which contribute to the aggregated quantities of a given 
    ##  raster water pixel not related to WSE, water surface area, water fraction or sigma0
    n_other_pix = dsv_ex['n_other_pix'][:]
    # Fraction of water_area covered by dark water. This value is typically between 0 and 1, 
    ##  with 0 indicating no dark water and 1 indicating 100% dark water. 
    ##  However, the value may be outside the range from 0 to 1 due to noise in the underlying area estimates.
    dark_frac = dsv_ex['dark_frac'][:]
    ice_clim_flag = dsv_ex['ice_clim_flag'][:]# Climatological ice cover flag: 0,1,2 = no ice, possible ice, fully covered
    ice_dyn_flag = dsv_ex['ice_dyn_flag'][:] #More reliable than ice_clim_flag when available.  Dynamic ice cover flag: 0,1,2 = no ice, possible ice, fully covered
    layover_impact = dsv_ex['layover_impact'][:] # Continuous value indicating an estimate of the systematic WSE error in meters due to layover
    # Geodetic longitude and latitude coordinates giving the 
    ## horizontal location of the center of the observed pixel. 
    ## The latitude is a geodetic latitude with respect to the 
    ## reference ellipsoid, which is defined by the semi_major_axis 
    ## and inverse_flattening attributes of the crs variable
    lon_all = dsv_ex['longitude'][:]
    lat_all = dsv_ex['latitude'][:]
    ikpLL = np.where((lat_all>=LLMM[0])&(lat_all<=LLMM[1])&(lon_all>=LLMM[2])&(lon_all<=LLMM[3]))
    ikpQF = np.where((wse_qual<=1)&(n_wse_pix>0))
    ikp = np.where((lat_all>=LLMM[0])&(lat_all<=LLMM[1])&(lon_all>=LLMM[2])&(lon_all<=LLMM[3])&(wse_qual<=1)&(n_wse_pix>0)) # (ice_clim_flag==0)& --> not accurate enough
    print(str(np.shape(ikpLL))+' measurements within '+str(LLMM))
    print(str(np.shape(ikpQF))+' measurements pass quality check')
    print(str(np.shape(ikp))+' total measurements pass')
    if np.size(ikp)>0:
        NC = {}
        # Variables
        crs = dsv_ex['crs'][:] # UTM zone coordinate reference system.
        NC['x'] = dsv_ex['x'][:] # easting UTM projection
        NC['y'] = dsv_ex['y'][:] # northing UTM projection

        NC['lon'] = lon_all[ikp[0],ikp[1]]
        NC['lat'] = lat_all[ikp[0],ikp[1]]
        # water surface elevation (relative to geoid) --> 
        ## corrections already applied: tropo, iono, crossover correction and tide.
        NC['wse'] = dsv_ex['wse'][:][ikp[0],ikp[1]]
        wse_qual_bitwise = dsv_ex['wse_qual_bitwise'][:][ikp[0],ikp[1]] # why the wse_qual flag is set as it is
        NC['wse_uncert'] = dsv_ex['wse_uncert'][:][ikp[0],ikp[1]] # 1-sigma uncertainty in the water surface elevation.
        # Water surface area estimate. The reported value is the total estimated water surface area within the pixel
        NC['water_area'] = dsv_ex['water_area'][:][ikp[0],ikp[1]] #water surface are
        water_area_qual_bitwise = dsv_ex['water_area_qual_bitwise'][:][ikp[0],ikp[1]]
        NC['water_area_uncert'] = dsv_ex['water_area_uncert'][:][ikp[0],ikp[1]] # 1-sigma uncertainty in the water surface area.
        # Water fraction estimate. The reported value is the total estimated water fraction within the pixel.
        ##  = water_area divided by the total pixel area
        NC['water_frac'] = dsv_ex['water_frac'][:][ikp[0],ikp[1]] # water fraction
        NC['water_frac_uncert'] = dsv_ex['water_frac_uncert'][:][ikp[0],ikp[1]] # 1-sigma uncertainty in the water fraction.
        # Normalized radar cross section (NRCS) or sigma0. This radar backscatter estimate
        ##  is given in linear units (not decibels). The value may be slightly negative due 
        ##  to errors in noise estimation and subtraction.
        NC['sig0'] = dsv_ex['sig0'][:][ikp[0],ikp[1]] # normalized radar cross section
        sig0_qual_bitwise = dsv_ex['sig0_qual_bitwise'][:][ikp[0],ikp[1]]
        NC['sig0_uncert'] = dsv_ex['sig0_uncert'][:][ikp[0],ikp[1]] # 1-sigma uncertainty in sigma0. The value is provided in linear units.
        # Incidence angle, which is the angle of the look vector with respect to the local “up”
        ##  direction where the look vector intersects with the reference DEM. 
        ##  The incidence angle is between 0 and 90  ̊.
        NC['inc'] = dsv_ex['inc'][:][ikp[0],ikp[1]]
        # Approximate cross-track location of the pixel aggregated from the reported values in the L2_HR_PIXC product. 
        ##  This value is reported as a signed distance to the right of the spacecraft nadir point; 
        ##  negative values indicate that the pixel is on the left side of the nadir track. 
        ##  The distance is computed using a local spherical Earth approximation and corresponds to the pixel 
        ##  reference location based on the reference digital elevation model (DEM), not the computed geolocation.
        NC['cross_track'] = dsv_ex['cross_track'][:][ikp[0],ikp[1]]
        # The illumination time of the pixel is the average measurement time of the contributing L2_HR_PIXC samples.
        ##  The variable illumination_time has an attribute tai_utc_difference, which represents the difference 
        ##  between TAI and UTC (i.e., total number of leap seconds) at the time of the first measurement record 
        ##  in the raster product.• illumination_time_tai[0] = illumination_time[0] + tai_utc_difference
        NC['t_utc'] = dsv_ex['illumination_time'][:][ikp[0],ikp[1]] # UTC time scale (seconds since January 1, 2000 00:00:00 UTC, which is equivalent to January 1, 2000 00:00:32 TAI)
        illumination_time_tai = dsv_ex['illumination_time_tai'][:][ikp[0],ikp[1]] # Pixel illumination time in TAI time scale (seconds since January 1, 2000 00:00:00 TAI, which is equivalent to December 31, 1999 23:59:28 UTC)
        # 2-way atmospheric radiometric correction to sig0
        ## remove this correction by a dividing the linear sig0 by sig0_cor_atmos_model, or the user can 
        ##  replace the correction by dividing it out and multiplying by a new correction.
        NC['sig0_cor_atmos_model'] = dsv_ex['sig0_cor_atmos_model'][:][ikp[0],ikp[1]] 
        # Height correction to wse
        # uncorrected pixel height = wse - height_cor_xover
        NC['height_cor_xover'] = dsv_ex['height_cor_xover'][:][ikp[0],ikp[1]]
        NC['geoid'] = dsv_ex['geoid'][:][ikp[0],ikp[1]]
        NC['solid_earth_tide'] = dsv_ex['solid_earth_tide'][:][ikp[0],ikp[1]]
        NC['load_tide_fes'] = dsv_ex['load_tide_fes'][:][ikp[0],ikp[1]]
        NC['load_tide_got'] = dsv_ex['load_tide_got'][:][ikp[0],ikp[1]]
        NC['pole_tide'] = dsv_ex['pole_tide'][:][ikp[0],ikp[1]]
        NC['model_dry_tropo_cor'] = dsv_ex['model_dry_tropo_cor'][:][ikp[0],ikp[1]]
        NC['model_wet_tropo_cor'] = dsv_ex['model_wet_tropo_cor'][:][ikp[0],ikp[1]]
        NC['iono_cor_gim_ka'] = dsv_ex['iono_cor_gim_ka'][:][ikp[0],ikp[1]]
        # applied already:height_cor_xover, (solid_earth_tide, load_tide_fes, pole_tide), model_dry_tropo_cor, model_wet_tropo_cor
        # not applied: 
    else:
        NC=0
    return NC
        
def pull_and_save_raster(fn_ex,LLMM,FILESV=[]):
    # Pull data
    FN = os.listdir(pth_temp)
    if fn_ex in FN:
        if 'SWOT_L2_HR_Raster' in fn_ex:
            print('Open file')
            dsv_ex= Dataset(pth_temp+fn_ex).variables
            NC = raster_var(dsv_ex,LLMM)
            if NC!=0:
                print('Saving '+FILESV)
                print('Data size: '+str(np.size(NC['geoid'])))
                root_grp = Dataset(FILESV, 'w', format='NETCDF4')
                root_grp.description = "This file contains SWOT_L2_HR_Raster_ ... data filtered for LLMM = "+str(LLMM)
                root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
                #Nx,Ny=np.size(NC['x']),np.size(NC['y'])
                Nx = np.size(NC['lat'])
                root_grp.createDimension('X', Nx)
                #root_grp.createDimension('Y', Ny)
                var = NC.keys()
                for vv in var:
                    if vv not in ['x','y']:
                        #print(vv)
                        #print('size :'+str(np.size(NC[vv])))
                        xi = root_grp.createVariable(vv, 'f8', ('X'))
                        xi[:] = NC[vv]
                root_grp.close()

#########################################################################################################################
#########################################################################################################################
############## RiverSP PRODUCT FUNCTIONS (Reach and Node) ############## 
def pull_and_save_lake(fn_zip,LLMM,FILESV=[]):
    Shapefile = geopandas.read_file(pth_temp+fn_zip)
    #print(list(Shapefile.keys()))
    latIn = Shapefile.lat.to_numpy() #Shapefile.lat.info(),Shapefile.lat.describe()
    lonIn = Shapefile.lon.to_numpy()
    ikp = np.where((latIn>=LLMM[0])&(latIn<=LLMM[1])&(lonIn>=LLMM[2])&(lonIn<=LLMM[3]))[0] #np.where((np.abs(latIn)<=90)&(np.abs(lonIn)<=360))[0]
    LIB={}
    print('Number of points within region of interest: '+str(np.size(ikp)))
    if np.size(ikp)>0:
        riverID = list(Shapefile.river_name[ikp])
        LIB['river_id']=alpha2num(riverID)
        lky = list(LIB.keys())
        Nk = np.shape(lky)[0]
        if np.size(FILESV)!=0:
            print('Save file :'+FILESV)
            root_grp = Dataset(FILESV, 'w', format='NETCDF4')
            root_grp.description = "This file contains SWOT_L2_HR_River ... data filtered for LLMM = "+str(LLMM)
            root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
            Nx = np.size(LIB['lat'])
            root_grp.createDimension('t', Nx)
            for ii in np.arange(Nk):
                xi = root_grp.createVariable(lky[ii], 'f8', ('t'))
                xi[:] = LIB[lky[ii]]
            root_grp.close()


def pull_and_save_node(fn_zip,LLMM,FILESV=[]):
    Shapefile = geopandas.read_file(pth_temp+fn_zip)
    #print(list(Shapefile.keys()))
    latIn = Shapefile.lat.to_numpy() #Shapefile.lat.info(),Shapefile.lat.describe()
    lonIn = Shapefile.lon.to_numpy()
    ikp = np.where((latIn>=LLMM[0])&(latIn<=LLMM[1])&(lonIn>=LLMM[2])&(lonIn<=LLMM[3]))[0] #np.where((np.abs(latIn)<=90)&(np.abs(lonIn)<=360))[0]
    LIB={}
    print('Number of points within region of interest: '+str(np.size(ikp)))
    if np.size(ikp)>0:
        riverID = list(Shapefile.river_name[ikp])
        LIB['river_id']=alpha2num(riverID)
        LIB['lat']=latIn[ikp]
        LIB['lon']=lonIn[ikp]
        LIB['lat_err'] = Shapefile.lat_u.to_numpy()[ikp]
        LIB['lon_err'] = Shapefile.lon_u.to_numpy()[ikp]
        LIB['t_utc'] = Shapefile.time.to_numpy()[ikp]
        LIB['t_tai'] = Shapefile.time_tai.to_numpy()[ikp]
        #t_str = Shapefile.time_str.to_list()[ikp]
        LIB['wse'] = Shapefile.wse.to_numpy()[ikp]
        LIB['wse_uncert'] = Shapefile.wse_u.to_numpy()[ikp]
        LIB['wid'] = Shapefile.width.to_numpy()[ikp]
        LIB['wid_uncert'] = Shapefile.width_u.to_numpy()[ikp]
        LIB['water_area'] = Shapefile.area_total.to_numpy()[ikp]
        LIB['water_area_uncert'] = Shapefile.area_tot_u.to_numpy()[ikp]
        LIB['dst'] = Shapefile.node_dist.to_numpy()[ikp]

        LIB['geoid'] = Shapefile.geoid_hght.to_numpy()[ikp]
        LIB['solid_earth_tide'] = Shapefile.solid_tide.to_numpy()[ikp]
        LIB['load_tide_fes'] = Shapefile.load_tidef.to_numpy()[ikp]
        LIB['load_tide_got'] = Shapefile.load_tideg.to_numpy()[ikp]
        LIB['pole_tide'] = Shapefile.pole_tide.to_numpy()[ikp]
        LIB['model_dry_tropo_cor'] = Shapefile.dry_trop_c.to_numpy()[ikp]
        LIB['model_wet_tropo_cor'] = Shapefile.wet_trop_c.to_numpy()[ikp]
        LIB['iono_cor_gim_ka'] = Shapefile.iono_c.to_numpy()[ikp]
        LIB['height_cor_xover'] = Shapefile.xovr_cal_c.to_numpy()[ikp]
        LIB['sig0'] = Shapefile.rdr_sig0.to_numpy()[ikp]
        LIB['sig0_uncert'] = Shapefile.rdr_sig0_u.to_numpy()[ikp]

        LIB['dark_frac'] = Shapefile.dark_frac.to_numpy()[ikp] # fraction of reach_area covered by dark water
        LIB['flag_ice_clim'] = Shapefile.ice_clim_f.to_numpy()[ikp] #(0=no ice)
        LIB['flag_ice_dyn'] = Shapefile.ice_dyn_f.to_numpy()[ikp] #(0=no ice)
        LIB['flow_angle'] = Shapefile.flow_angle.to_numpy()[ikp] # river flow direction (0-360 deg)
        LIB['quality_node'] = Shapefile.node_q.to_numpy()[ikp] #(0=good)
        LIB['n_good_pix'] = Shapefile.n_good_pix.to_numpy()[ikp] # number of pixels that have valid wse
        LIB['quality_xovr_cal'] = Shapefile.xovr_cal_q.to_numpy()[ikp] # number of pixels that have valid wse
        lky = list(LIB.keys())
        Nk = np.shape(lky)[0]
        if np.size(FILESV)!=0:
            print('Save file :'+FILESV)
            root_grp = Dataset(FILESV, 'w', format='NETCDF4')
            root_grp.description = "This file contains SWOT_L2_HR_River ... data filtered for LLMM = "+str(LLMM)
            root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
            Nx = np.size(LIB['lat'])
            root_grp.createDimension('t', Nx)
            for ii in np.arange(Nk):
                xi = root_grp.createVariable(lky[ii], 'f8', ('t'))
                xi[:] = LIB[lky[ii]]
            root_grp.close()

def pull_and_save_reach(fn_zip,LLMM,FILESV=[]):
    Shapefile = geopandas.read_file(pth_temp+fn_zip)
    latIn = Shapefile.p_lat.to_numpy() #Shapefile.lat.info(),Shapefile.lat.describe()
    lonIn = Shapefile.p_lon.to_numpy()
    ikp = np.where((latIn>=LLMM[0])&(latIn<=LLMM[1])&(lonIn>=LLMM[2])&(lonIn<=LLMM[3]))[0] #np.where((np.abs(latIn)<=90)&(np.abs(lonIn)<=360))[0]
    LIB={}
    print('Number of points within region of interest: '+str(np.size(ikp)))
    if np.size(ikp)>0:
        #uRiv,LIB['river_id'] = np.unique(list(Shapefile.river_name[ikp]),return_inverse=True)
        riverID = list(Shapefile.river_name[ikp])
        LIB['river_id']=alpha2num(riverID)
        LIB['lat']=latIn[ikp]
        LIB['lon']=lonIn[ikp]
        LIB['t_utc'] = Shapefile.time.to_numpy()[ikp]
        LIB['t_tai'] = Shapefile.time_tai.to_numpy()[ikp]
        #t_str = Shapefile.time_str.to_list()[ikp]
        LIB['wse'] = Shapefile.wse.to_numpy()[ikp]
        LIB['wse_uncert'] = Shapefile.wse_u.to_numpy()[ikp]
        LIB['wid'] = Shapefile.width.to_numpy()[ikp]
        LIB['wid_uncert'] = Shapefile.width_u.to_numpy()[ikp]
        LIB['water_area'] = Shapefile.area_total.to_numpy()[ikp]
        LIB['water_area_uncert'] = Shapefile.area_tot_u.to_numpy()[ikp]

        LIB['p_length'] = Shapefile.p_length.to_numpy()[ikp]
        LIB['p_width'] = Shapefile.p_width.to_numpy()[ikp]
        LIB['node_dist'] = Shapefile.node_dist.to_numpy()[ikp]
        LIB['layovr_val'] = Shapefile.layovr_val.to_numpy()[ikp]
        LIB['p_n_nodes'] = Shapefile.p_n_nodes.to_numpy()[ikp]
        LIB['p_low_slp'] = Shapefile.p_low_slp.to_numpy()[ikp]

        '''
        A positive slope means that the downstream WSE is lower. The slope is not provided for connected lakes (T = 3 in the reach_id attribute)
        '''
        LIB['slope'] = Shapefile.slope.to_numpy()[ikp] #[m/m]
        LIB['slope_uncert'] = Shapefile.slope_u.to_numpy()[ikp]#[m/m]

        LIB['slope2'] = Shapefile.slope2.to_numpy()[ikp]#[m/m]
        LIB['slope2_uncert'] = Shapefile.slope2_u.to_numpy()[ikp]#[m/m]
        
        LIB['dst'] = Shapefile.node_dist.to_numpy()[ikp]

        LIB['geoid'] = Shapefile.geoid_hght.to_numpy()[ikp]
        LIB['geoid_slope'] = Shapefile.geoid_slop.to_numpy()[ikp]
        LIB['solid_earth_tide'] = Shapefile.solid_tide.to_numpy()[ikp]
        LIB['load_tide_fes'] = Shapefile.load_tidef.to_numpy()[ikp]
        LIB['load_tide_got'] = Shapefile.load_tideg.to_numpy()[ikp]
        LIB['pole_tide'] = Shapefile.pole_tide.to_numpy()[ikp]
        LIB['model_dry_tropo_cor'] = Shapefile.dry_trop_c.to_numpy()[ikp]
        LIB['model_wet_tropo_cor'] = Shapefile.wet_trop_c.to_numpy()[ikp]
        LIB['iono_cor_gim_ka'] = Shapefile.iono_c.to_numpy()[ikp]
        LIB['height_cor_xover'] = Shapefile.xovr_cal_c.to_numpy()[ikp]


        LIB['dark_frac'] = Shapefile.dark_frac.to_numpy()[ikp] # fraction of reach_area covered by dark water
        LIB['flag_ice_clim'] = Shapefile.ice_clim_f.to_numpy()[ikp] #(0=no ice)
        LIB['flag_ice_dyn'] = Shapefile.ice_dyn_f.to_numpy()[ikp] #(0=no ice)
        LIB['quality_reach'] = Shapefile.reach_q.to_numpy()[ikp] #(0=good)
        LIB['n_good_nod'] = Shapefile.n_good_nod.to_numpy()[ikp] # number of pixels that have valid wse
        LIB['quality_xovr_cal'] = Shapefile.xovr_cal_q.to_numpy()[ikp] # number of pixels that have valid wse
        lky = list(LIB.keys())
        Nk = np.shape(lky)[0]
        if np.size(FILESV)!=0:
            print('Save file :'+FILESV)
            root_grp = Dataset(FILESV, 'w', format='NETCDF4')
            root_grp.description = "This file contains SWOT_L2_HR_River ... data filtered for LLMM = "+str(LLMM)
            root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
            Nx = np.size(LIB['lat'])
            root_grp.createDimension('t', Nx)
            for ii in np.arange(Nk):
                xi = root_grp.createVariable(lky[ii], 'f8', ('t'))
                xi[:] = LIB[lky[ii]]
            root_grp.close()



#########################################################################################################################
#########################################################################################################################
####### OCEAN FUNCTIONS
#########################################################################################################################
#########################################################################################################################

#########################################################################################################################
#########################################################################################################################
############## L2 LR Expert (2 km x 2 km) PRODUCT FUNCTIONS ############## 

def karin_expert_filt(dsv,FLAG_SOURCE=[],LLMM=[]):
    # dsv,FLAG_SOURCE=dsv_ex,[]
    # SSH flags
    ssh_karin_qual=dsv['ssh_karin_qual'][:] #>134217728 or 268435456==bad
    ssh_karin_qual[ssh_karin_qual<268435456]=0
    ssh_karin_qual[ssh_karin_qual>=268435456]=1
    ssha_karin_qual=dsv['ssha_karin_qual'][:]
    ssha_karin_qual[ssha_karin_qual<268435456]=0
    ssha_karin_qual[ssha_karin_qual>=268435456]=1
    ssh_karin_2_qual=dsv['ssh_karin_2_qual'][:]
    ssh_karin_2_qual[ssh_karin_2_qual<268435456]=0
    ssh_karin_2_qual[ssh_karin_2_qual>=268435456]=1
    ssha_karin_2_qual=dsv['ssha_karin_2_qual'][:]
    ssha_karin_2_qual[ssha_karin_2_qual<268435456]=0
    ssha_karin_2_qual[ssha_karin_2_qual>=268435456]=1
    # Parameter flags
    swh_karin_qual=dsv['swh_karin_qual'][:] #>=131072 == bad
    swh_karin_qual[swh_karin_qual<131072]=0
    swh_karin_qual[swh_karin_qual>=131072]=1
    sig0_karin_qual=dsv['sig0_karin_qual'][:]#>65536 ==bad
    sig0_karin_qual[sig0_karin_qual<65536]=0
    sig0_karin_qual[sig0_karin_qual>=65536]=1
    sig0_karin_2_qual=dsv['sig0_karin_2_qual'][:]#>65536 ==bad
    sig0_karin_2_qual[sig0_karin_2_qual<65536]=0
    sig0_karin_2_qual[sig0_karin_2_qual>=65536]=1
    ws_karin_qual=dsv['wind_speed_karin_qual'][:]#>65536 ==bad
    ws_karin_qual[ws_karin_qual<65536]=0
    ws_karin_qual[ws_karin_qual>=65536]=1
    ws_karin_2_qual=dsv['wind_speed_karin_2_qual'][:]#>65536 ==bad
    ws_karin_2_qual[ws_karin_2_qual<65536]=0
    ws_karin_2_qual[ws_karin_2_qual>=65536]=1
    # Classifications
    ancillary_surface_classification_flag = dsv['ancillary_surface_classification_flag'][:] #0 =open_ocean
    ancillary_surface_classification_flag[ancillary_surface_classification_flag>0]=1
    dynamic_ice_flag = dsv['dynamic_ice_flag'][:] # 0 =no_ice
    dynamic_ice_flag[dynamic_ice_flag>0]=1
    rain_flag = dsv['rain_flag'][:] #0 =no_rain
    rain_flag[rain_flag>0]=1
    rad_surf_type = dsv['rad_surface_type_flag'][:]#(9865, 2) #0=open_ocean, 1=coastal_ocean
    open_coast = np.copy(rad_surf_type)
    open_coast[open_coast>1]=1
    # Orbit flag
    orbit_qual = dsv['orbit_qual'][:]#0=good
    orbit_qual[orbit_qual>0]=1
    # Flag indicating the quality of the height correction from crossover calibration.
    height_cor_xover_qual = dsv['height_cor_xover_qual'][:] #[0 1 2]=good suspect bad
    # Parameter source flgs
    flag_para = 0
    flag_para2 = 0
    ### source flag for SWH (nadir, karin or model)
    swh_ws_karin_source = dsv['swh_wind_speed_karin_source'][:] #(9865, 69) # [1 2 4] = nadir_altimeter karin model
    swh_ws_karin_source_2 = dsv['swh_wind_speed_karin_source_2'][:]#(9865, 69) # [1 2 4] = nadir_altimeter karin model
    ### source flags for SWH and WS in SSB (nadir, karin or model)
    swh_ssb_cor_source = dsv['swh_ssb_cor_source'][:]#(9865, 69) source flag for SWH (nadir, karin or model)
    swh_ssb_cor_source_2 = dsv['swh_ssb_cor_source_2'][:]#(9865, 69) source flag for SWH (nadir, karin or model)
    ws_ssb_cor_source = dsv['wind_speed_ssb_cor_source'][:]#(9865, 69) source flag for WS (nadir, karin or model)
    ws_ssb_cor_source_2 = dsv['wind_speed_ssb_cor_source_2'][:]#(9865, 69) source flag for WS (nadir, karin or model)
    if np.size(FLAG_SOURCE)!=0:
            flag_para=np.ones(np.shape(swh_ws_karin_source))
            flag_para[flag_para==FLAG_SOURCE]=0
            flag_para2=np.ones(np.shape(swh_ws_karin_source))
            flag_para2[flag_para2==FLAG_SOURCE]=0
    # Bounding box filter
    if np.size(LLMM)!=0:
        lat = dsv['latitude'][:]#np.nanmean(dsv['latitude'][:],axis=1)#(9865, 69) latitude (positive N, negative S)
        lon = lon360_to_lon180(dsv['longitude'][:])#np.nanmean(dsv['longitude'][:],axis=1))
        buff=0.1
        illmm_geo = np.where((lat>=LLMM[0]-buff)&(lat<=LLMM[1]+buff)&(lon>=LLMM[2]-buff)&(lon<=LLMM[3]+buff))
        uniqT = np.unique(illmm_geo[0])
        if np.size(uniqT)!=0:
            print('illmm_geo',illmm_geo)
            print('uniqT',uniqT)
            illmm = np.arange(np.nanmin(uniqT),np.nanmax(uniqT)+1)
            '''
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(lon,lat,'.',color='green')
            plt.plot(np.asarray(LLMM[2:]),np.asarray(LLMM[:2]),'o',color='black')
            plt.plot(lon[illmm,:],lat[illmm,:],'.',color='red')
            '''
    else:
        illmm = np.arange(np.shape(ssh_karin_qual)[0])
        uniqT = np.arange(np.shape(lon)[0])
    #combine flags
    if np.size(uniqT)!=0:
        flag = ws_karin_qual+sig0_karin_qual+ssh_karin_qual+ssha_karin_qual+flag_para
        flag2 = ws_karin_2_qual+sig0_karin_2_qual+ssh_karin_2_qual+ssha_karin_2_qual+flag_para2
        flags_non12 = swh_karin_qual+ancillary_surface_classification_flag+dynamic_ice_flag+rain_flag
        ikeep = np.copy(illmm)#np.where(orbit_qual==0)[0]
        flagsi=(flag[ikeep,:]+flags_non12[ikeep,:])#.astype(int)
        flags2i=(flag2[ikeep,:]+flags_non12[ikeep,:])#.astype(int)
        flags = np.empty(np.shape(flagsi))*np.nan
        flags2 = np.empty(np.shape(flags2i))*np.nan
        flags[flagsi==0]=0.000
        flags2[flags2i==0]=0.000
        flags = np.zeros(np.shape(flagsi))
        flags2 = np.zeros(np.shape(flags2i))
    else:
        ikeep,flags,flags2,rad_surf_type=[],[],[],[]
    return ikeep,flags,flags2,rad_surf_type

def pull_expert_other(dsv,iflt,flags):
    # num_pt_avg,distance_to_coast,heading_to_coast,depth_or_elevation=pull_expert_other(dsv,iflt,flags)
    polarization_karin = dsv['polarization_karin'][:][iflt,:]#(9865, 2)
    num_pt_avg = dsv['num_pt_avg'][:][iflt,:]+flags#(9865, 69)
    distance_to_coast = dsv['distance_to_coast'][:][iflt,:]+flags#(9865, 69)
    heading_to_coast = dsv['heading_to_coast'][:][iflt,:]+flags#(9865, 69)
    doppler_centroid = dsv['doppler_centroid'][:][iflt,:]+flags#(9865, 69)
    phase_bias_ref_surface = dsv['phase_bias_ref_surface'][:][iflt,:]+flags#(9865, 69)
    obp_ref_surface = dsv['obp_ref_surface'][:][iflt,:]+flags#(9865, 69)
    depth_or_elevation = dsv['depth_or_elevation'][:][iflt,:]+flags#(9865, 69)
    rain_rate = dsv['rain_rate'][:][iflt,:]+flags#(9865, 69)
    ice_conc = dsv['ice_conc'][:][iflt,:]+flags#(9865, 69)
    return num_pt_avg,distance_to_coast,heading_to_coast,depth_or_elevation

def pull_expert_orbit(dsv,iflt):
    # sc_alt,xtrack_angle,velocity_heading = pull_expert_orbit(dsv,iflt)
    sc_alt = dsv['sc_altitude'][:][iflt] #(9865)
    orbit_alt_rate = dsv['orbit_alt_rate'][:][iflt] #(9865)
    sc_roll = dsv['sc_roll'][:][iflt] #(9865)
    sc_pitch = dsv['sc_pitch'][:][iflt] #(9865)
    sc_yaw = dsv['sc_yaw'][:][iflt] #(9865)
    #Angle with respect to true north of the cross-track direction to the right of the spacecraft velocity vector.
    xtrack_angle = dsv['cross_track_angle'][:][iflt] #(9865)[degrees]
    # heading of the spacecraft Earth-relative velocity vector
    velocity_heading = dsv['velocity_heading'][:][iflt] #(9865) [degrees]
    return sc_alt,xtrack_angle,velocity_heading

def pull_expert_radiometer(dsv,iflt):
    rad_tmb_187 = dsv['rad_tmb_187'][:][iflt] #(9865, 2)
    rad_tmb_238 = dsv['rad_tmb_238'][:][iflt] #(9865, 2)
    rad_tmb_340 = dsv['rad_tmb_340'][:][iflt] #(9865, 2)
    rad_water_vapor = dsv['rad_water_vapor'][:][iflt] #(9865, 2)
    rad_cloud_liquid_water = dsv['rad_cloud_liquid_water'][:][iflt] #(9865, 2)
    ws_rad = dsv['wind_speed_rad'][:][iflt,:] #(9865, 2)
    return ws_rad

def pull_expert_waves_and_ssb(dsv,iflt,flags):
    # swh_karin,swh_nadir,swh_model,mwave_dir,mwave_t02,ssb_cor,ssb_cor_2=pull_expert_waves_and_ssb(dsv,iflt,flags)
    # SWH
    #Significant wave height from KaRIn volumetric correlation.
    swh_karin = dsv['swh_karin'][:][iflt,:]+flags #(9865, 69)
    # significant wave height from nadir altimeter
    swh_nadir = dsv['swh_nadir_altimeter'][:][iflt,:]+flags #(9865, 69)
    # significant wave height from wave model (ECMWF))
    swh_model = dsv['swh_model'][:][iflt,:]+flags #(9865, 69)
    mwave_dir = dsv['mean_wave_direction'][:][iflt,:]+flags#(9865, 69) [degrees] Meteo France Wave Model (MF-WAM)
    mwave_t02 = dsv['mean_wave_period_t02'][:][iflt,:]+flags#(9865, 69) [degrees] Meteo France Wave Model (MF-WAM)
    #SSB
    ssb_cor = dsv['sea_state_bias_cor'][:][iflt,:]+flags #(9865, 69) CNES (dependent on wind_speed_karin)
    ssb_cor_2 = dsv['sea_state_bias_cor_2'][:][iflt,:]+flags#(9865, 69) CNES (dependent on wind_speed_karin_2)
    return swh_karin,swh_nadir,swh_model,mwave_dir,mwave_t02,ssb_cor,ssb_cor_2

def pull_expert_ssha(dsv,iflt,flags):
    # utc,lat,lon,ssh_karin,ssh_karin_2 = pull_expert_ssha(dsv,iflt,flags)
    utc = dsv['time'][:][iflt] #(9865) #seconds since 2000-01-01 00:00:00.0
    tai = dsv['time_tai'][:][iflt] #(9865) #seconds since 2000-01-01 00:00:00.0
    tai_utc_difference=dsv['time'].tai_utc_difference
    lat = dsv['latitude'][:][iflt,:]+flags#(9865, 69) latitude (positive N, negative S)
    lon = dsv['longitude'][:][iflt,:]+flags #(9865, 69) [0-360]
    lat_nadir = dsv['latitude_nadir'][:][iflt] #(9865)
    lon_nadir = dsv['longitude_nadir'][:][iflt] #(9865)  [0-360]

    ssh_karin = dsv['ssh_karin'][:][iflt,:]+flags #(9865, 69)
    # SSHA KaRIn measurement = ssh_karin - mean_sea_surface_cnescls - solid_earth_tide - ocean_tide_fes – internal_tide_hret - pole_tide - dac.
    ssha_karin = dsv['ssha_karin'][:][iflt,:]+flags #(9865, 69)
    ssh_karin_2 = dsv['ssh_karin_2'][:][iflt,:]+flags #(9865, 69)
    # SSHA KaRIn measurement = ssh_karin_2 - mean_sea_surface_cnescls - solid_earth_tide - ocean_tide_fes – internal_tide_hret - pole_tide - dac.
    ssha_karin_2 = dsv['ssha_karin_2'][:][iflt,:]+flags #(9865, 69)
    """Height correction from crossover calibration. 
    To apply this correction the value of height_cor_xover should be 
    added to the value of ssh_karin, ssh_karin_2, ssha_karin, and ssha_karin_2."""
    height_cor_xover = dsv['height_cor_xover'][:][iflt,:]+flags
    
    sig0_karin = dsv['sig0_karin'][:][iflt,:]+flags #(9865, 69)
    sig0_karin_2 = dsv['sig0_karin_2'][:][iflt,:]+flags #(9865, 69)
    ws_karin = dsv['wind_speed_karin'][:][iflt,:]+flags #(9865, 69)
    ws_karin_2 = dsv['wind_speed_karin_2'][:][iflt,:]+flags #(9865, 69)
    ws_model_u = dsv['wind_speed_model_u'][:][iflt,:]+flags #(9865, 69)
    ws_model_v = dsv['wind_speed_model_v'][:][iflt,:]+flags #(9865, 69)
    sig0_cor_atmos_model = dsv['sig0_cor_atmos_model'][:][iflt,:]+flags #(9865, 69) sig0_cor_atmos_model is already applied in computing sig0_karin_2
    sig0_cor_atmos_rad = dsv['sig0_cor_atmos_rad'][:][iflt,:]+flags #(9865, 69) sig0_cor_atmos_rad is already applied in computing sig0_karin
    return utc,lat,lon,ssh_karin,ssh_karin_2,ssha_karin,ssha_karin_2,height_cor_xover

def pull_expert_geophysical(dsv,iflt,flags,MSS='cnes15',OCEAN_TIDE='fes14b',INT_TIDE='hret',DAC=True):
    # mss,tE,tO,int_tide,tP,dac,Td_model,Tw,Tw_model,iono_gim_ka,geoid,mdt = pull_expert_geophysical(dsv,iflt,flags,MSS='cnes15',OCEAN_TIDE='fes14b',INT_TIDE='hret',DAC=True,RAD=True)
    geoid = dsv['geoid'][:][iflt,:]+flags #(9865, 69)
    mdt = dsv['mean_dynamic_topography'][:][iflt,:]+flags #(9865, 69)
    mdt_uncert = dsv['mean_dynamic_topography_uncert'][:][iflt,:]+flags #(9865, 69)
    tE = dsv['solid_earth_tide'][:][iflt,:]+flags #(9865, 69)
    tP = dsv['pole_tide'][:][iflt,:]+flags #(9865, 69)
    Td_model = dsv['model_dry_tropo_cor'][:][iflt,:]+flags #(9865, 69)
    iono_gim_ka = dsv['iono_cor_gim_ka'][:][iflt,:]+flags #(9865, 69)
    # Wet troposphere correction
    Tw = dsv['rad_wet_tropo_cor'][:][iflt,:]+flags #(9865, 69)
    Tw_model = dsv['model_wet_tropo_cor'][:][iflt,:]+flags #(9865, 69)
    # MSS 
    if MSS=='cnes15':
        mss = dsv['mean_sea_surface_cnescls'][:][iflt,:]+flags #(9865, 69) CNES_CLS_15
        mss_uncert = dsv['mean_sea_surface_cnescls_uncert'][:][iflt,:]+flags #(9865, 69)
    elif MSS=='dtu18':
        mss = dsv['mean_sea_surface_dtu'][:][iflt,:]+flags #(9865, 69) DTU18
        mss_uncert = dsv['mean_sea_surface_dtu_uncert'][:][iflt,:]+flags #(9865, 69)
    # Geocentric ocean tide height. Includes the sum total of the ocean tide, the corresponding load tide (load_tide_fes) and equilibrium long-period ocean tide height (ocean_tide_eq).
    if OCEAN_TIDE=='fes14b':
        tO = dsv['ocean_tide_fes'][:][iflt,:]+flags #(9865, 69) FES2014b (Carrere et al., 2016)
        tL = dsv['load_tide_fes'][:][iflt,:]+flags #(9865, 69) FES2014b (Carrere et al., 2016)
    elif OCEAN_TIDE=='got410c':
        tO = dsv['ocean_tide_got'][:][iflt,:]+flags #(9865, 69) GOT4.10c (Ray, 2013)
        tL = dsv['load_tide_got'][:][iflt,:]+flags #(9865, 69) GOT4.10c (Ray, 2013)
    tO_eq = dsv['ocean_tide_eq'][:][iflt,:]+flags #(9865, 69) # already applied to tO
    tO_non_eq = dsv['ocean_tide_non_eq'][:][iflt,:]+flags #(9865, 69)
    # Internal tide
    if INT_TIDE=='hret':
        int_tide = dsv['internal_tide_hret'][:][iflt,:]+flags #(9865, 69) #coherent internal tide (HRET), Zaron (2019)
    elif INT_TIDE=='sol2':
        int_tide = dsv['internal_tide_sol2'][:][iflt,:]+flags #(9865, 69) #coherent internal tide (Model 2)
    # Ocean response to atmosphere pressure changes
    if DAC==True:
        dac = dsv['dac'][:][iflt,:]+flags #(9865, 69)
    else:
        dac = dsv['inv_bar_cor'][:][iflt,:]+flags #(9865, 69)
    return mss,tE,tO,int_tide,tP,dac,Td_model,Tw,Tw_model,iono_gim_ka,geoid,mdt


#########################################################################################################################
#########################################################################################################################
############## L2 LR Unsmoothed (250 m x 250 m) PRODUCT FUNCTIONS ############## 
def pull_unsmooth_side(dsv_side_us):
    # dsv_side_us = dsv_right_us
    utc=dsv_side_us['time'][:] #(82323,)
    tai=dsv_side_us['time_tai'][:] #(82323,)
    tai_utc_difference=dsv_side_us['time'].tai_utc_difference
    lat=dsv_side_us['latitude'][:] #(82323, 240)
    lon=dsv_side_us['longitude'][:]#(82323, 240)
    lat_uncert=dsv_side_us['latitude_uncert'][:]#(82323, 240)
    lon_uncert=dsv_side_us['longitude_uncert'][:]#(82323, 240)
    polarization_karin=dsv_side_us['polarization_karin'][:]#(82323, 240)
    ssh_karin_2=dsv_side_us['ssh_karin_2'][:]#(82323, 240)
    ssh_karin_uncert=dsv_side_us['ssh_karin_uncert'][:]#(82323, 240)
    sig0_karin_2=dsv_side_us['sig0_karin_2'][:]#(82323, 240)
    sig0_karin_uncert=dsv_side_us['sig0_karin_uncert'][:]#(82323, 240)
    total_coherence=dsv_side_us['total_coherence'][:]#(82323, 240) #Total KaRIn interferometric coherence.
    mss=dsv_side_us['mean_sea_surface_cnescls'][:]#(82323, 240) # CNES_CLS_15
    #Center-beam 250 meter resolution power from KaRIn in real, linear units (not decibels).
    miti_power_250m=dsv_side_us['miti_power_250m'][:]#(82323, 240)
    #Center-beam 250 meter resolution power variance from KaRIn in real, linear units (not decibels).
    miti_power_var_250m=dsv_side_us['miti_power_var_250m'][:]#(82323, 240)
    # FIlter
    ssh_karin_2_qual=dsv_side_us['ssh_karin_2_qual'][:]#(82323, 240)
    sig0_karin_2_qual=dsv_side_us['sig0_karin_2_qual'][:]#(82323, 240)
    ancillary_surface_classification_flag=dsv_side_us['ancillary_surface_classification_flag'][:]#(82323, 240) # 0 =open ocean
    lonM = lon.filled(np.nan) 
    latM = lat.filled(np.nan) 
    ikeep = np.arange(np.size(utc))#np.where((lonM>=0)&(lonM<360))[0] #np.where(ancillary_surface_classification_flag==0)[0]
    return ikeep,utc,latM,lonM,ssh_karin_2,sig0_karin_2,mss

def pull_unsmooth(ds_us,LLMM,minT,maxT):
    dsv_right_us=ds_us['right'].variables
    dsv_left_us=ds_us['left'].variables
    ikeep_r,utc_r,lat_r,lon_r,ssh_karin_2_r,sig0_karin_2_r,mss_r = pull_unsmooth_side(dsv_right_us)
    ikeep_l,utc_l,lat_l,lon_l,ssh_karin_2_l,sig0_karin_2_l,mss_l = pull_unsmooth_side(dsv_left_us)
    #lon_us = np.hstack((lon_l,lon_r))#np.hstack((lon_l[ikeep_us,:],lon_r[ikeep_us,:]))
    #lat_us = np.hstack((lat_l,lat_r))
    #buff=1
    #illmm_geo = np.where((lat_us>LLMM[0]-buff)&(lat_us<LLMM[1]+buff)&(lon_us>LLMM[2]-buff)&(lon_us<LLMM[3]+buff))
    #uniqT = np.unique(illmm_geo[0])
    utc_usi = np.hstack((utc_l,utc_r))
    illmm = np.where((utc_usi>=minT)&(utc_usi<=maxT))[0]# np.arange(np.nanmin(uniqT),np.nanmax(uniqT)+1)
    ikeep_us = illmm#np.intersect1d(illmm,np.intersect1d(ikeep_r,ikeep_l))
    print('minT = '+str(np.round(minT,3))+', maxT = '+str(np.round(maxT,3)))
    print('ikeep_us: ',ikeep_us)
    utc_us = np.copy(utc_usi)[ikeep_us]
    lon_us = np.vstack((lon_l,lon_r))[ikeep_us,:]#np.hstack((lon_l[ikeep_us,:],lon_r[ikeep_us,:]))
    lat_us = np.vstack((lat_l,lat_r))[ikeep_us,:]
    ssh_karin_2_us = np.vstack((ssh_karin_2_l,ssh_karin_2_r))[ikeep_us,:]
    sig0_karin_2_us = np.vstack((sig0_karin_2_l,sig0_karin_2_r))[ikeep_us,:]
    mss_us = np.vstack((mss_l,mss_r))[ikeep_us,:]
    return ikeep_us,utc_us,lat_us,lon_us,ssh_karin_2_us,sig0_karin_2_us,mss_us

def pull_expert_and_unsmoothed_swot(fn_ex,fn_us,LLMM,FILESV=[]):
    # Pull data
    # https://podaac.github.io/tutorials/quarto_text/SWOT.html
    # podaac-data-subscriber -c SWOT_L2_LR_SSH_1.1 -d ./data/SWOT_L2_LR_SSH_1.1 --start-date 2022-12-16T00:00:00Z
    FN = os.listdir(pth_temp)
    if fn_ex in FN and fn_us in FN:
        dsv_ex= Dataset(pth_temp+fn_ex).variables
        ds_us= Dataset(pth_temp+fn_us)

        iflt,flags,flags2,rad_surf_type = karin_expert_filt(dsv_ex,FLAG_SOURCE=[],LLMM=LLMM)
        if np.size(iflt)==0:
            print(fn_ex+' is unavailable for '+str(LLMM))
        else:
            num_pt_avg,distance_to_coast,heading_to_coast,depth_or_elevation=pull_expert_other(dsv_ex,iflt,flags)
            sc_alt,xtrack_angle,velocity_heading = pull_expert_orbit(dsv_ex,iflt)
            swh_karin,swh_nadir,swh_model,mwave_dir,mwave_t02,ssb_cor,ssb_cor_2=pull_expert_waves_and_ssb(dsv_ex,iflt,flags)
            utc,lat,lon,ssh_karin,ssh_karin_2,ssha_karin,ssha_karin_2,height_cor_xover = pull_expert_ssha(dsv_ex,iflt,flags)
            mss,tE,tO,int_tide,tP,dac,Td_model,Tw,Tw_model,iono_gim_ka,geoid,mdt = pull_expert_geophysical(dsv_ex,iflt,flags,MSS='cnes15',OCEAN_TIDE='got410c',INT_TIDE='hret',DAC=True)
            ssha_karin_cor=ssha_karin+height_cor_xover
            ssha_karin_2_cor=ssha_karin_2+height_cor_xover
            pix = np.arange(np.shape(mss)[1])

            minT,maxT = np.nanmin(utc),np.nanmax(utc)
            iUS,utc_us,lat_us,lon_us,ssh_karin_2_us,sig0_karin_2_us,mss_us = pull_unsmooth(ds_us,LLMM,minT,maxT)

            # # SSHA KaRIn measurement = ssh_karin - mean_sea_surface_cnescls - solid_earth_tide - ocean_tide_fes – internal_tide_hret - pole_tide - dac.
            tE_us = interpolate_data(tE,lat,lon,lat_us,lon_us)
            tP_us = interpolate_data(tP,lat,lon,lat_us,lon_us)
            tO_us = interpolate_data(tO,lat,lon,lat_us,lon_us)
            dac_us = interpolate_data(dac,lat,lon,lat_us,lon_us)
            int_tide_us = interpolate_data(int_tide,lat,lon,lat_us,lon_us)
            height_cor_xover_us = interpolate_data(height_cor_xover,lat,lon,lat_us,lon_us)
            ssb_cor_2_us = interpolate_data(ssb_cor_2,lat,lon,lat_us,lon_us)
            ssha_karin_2_us = ssh_karin_2_us-mss_us-tE_us-tO_us-int_tide_us-tP_us-dac_us
            ssha_karin_2_cor_us = ssha_karin_2_us+height_cor_xover_us

            lon180 = lon360_to_lon180(lon)
            lon180_us = lon360_to_lon180(lon_us)
            if np.size(FILESV) > 0:
                print('Saving '+FILESV)
                root_grp = Dataset(FILESV, 'w', format='NETCDF4')
                root_grp.description = "This file contains SWOT_L2_LR_SSH_ ... data filtered for LLMM = "+str(LLMM)
                root_grp.history = "Author: "+AUTHOR+". Institute: "+INSTITUTION+" "+ time.ctime(time.time())
                Ntm,Npix=np.shape(ssha_karin_2_cor)
                Ntm_us,Npix_us=np.shape(ssha_karin_2_cor_us)
                root_grp.createDimension('t_ex', Ntm)
                root_grp.createDimension('pix_ex', Npix)
                root_grp.createDimension('t_us', Ntm_us)
                root_grp.createDimension('pix_us', Npix_us)
                var_ex = ['ssha_karin_cor','ssha_karin_2_cor','mss','dac','tO','tP','tE','int_tide','height_cor_xover',
                        'iono_gim_ka','Td_model','tW']
                xi = root_grp.createVariable('utc', 'f8', ('t_ex',))
                xi[:] = utc
                xi = root_grp.createVariable('xtrack_angle', 'f8', ('t_ex',))
                xi[:] = xtrack_angle
                xi = root_grp.createVariable('velocity_heading', 'f8', ('t_ex',))
                xi[:] = velocity_heading
                xi = root_grp.createVariable('lat', 'f8', ('t_ex','pix_ex'))
                xi[:] = lat
                xi = root_grp.createVariable('lon', 'f8', ('t_ex','pix_ex'))
                xi[:] = lon180
                xi = root_grp.createVariable('ssha_karin_cor', 'f8', ('t_ex','pix_ex'))
                xi[:] = ssha_karin_cor
                xi = root_grp.createVariable('ssha_karin_2_cor', 'f8', ('t_ex','pix_ex'))
                xi[:] = ssha_karin_2_cor
                xi = root_grp.createVariable('dac', 'f8', ('t_ex','pix_ex'))
                xi[:] = dac
                xi = root_grp.createVariable('tO', 'f8', ('t_ex','pix_ex'))
                xi[:] = tO
                xi = root_grp.createVariable('mss', 'f8', ('t_ex','pix_ex'))
                xi[:] = mss
                xi = root_grp.createVariable('int_tide', 'f8', ('t_ex','pix_ex'))
                xi[:] = int_tide
                xi = root_grp.createVariable('tP', 'f8', ('t_ex','pix_ex'))
                xi[:] = tP
                xi = root_grp.createVariable('tE', 'f8', ('t_ex','pix_ex'))
                xi[:] = tE
                xi = root_grp.createVariable('height_cor_xover', 'f8', ('t_ex','pix_ex'))
                xi[:] = height_cor_xover
                xi = root_grp.createVariable('iono_gim_ka', 'f8', ('t_ex','pix_ex'))
                xi[:] = iono_gim_ka
                xi = root_grp.createVariable('Td_model', 'f8', ('t_ex','pix_ex'))
                xi[:] = Td_model
                xi = root_grp.createVariable('Tw_model', 'f8', ('t_ex','pix_ex'))
                xi[:] = Tw_model
                xi = root_grp.createVariable('Tw', 'f8', ('t_ex','pix_ex'))
                xi[:] = Tw
                xi = root_grp.createVariable('ssb_cor', 'f8', ('t_ex','pix_ex'))
                xi[:] = ssb_cor
                xi = root_grp.createVariable('ssb_cor_2', 'f8', ('t_ex','pix_ex'))
                xi[:] = ssb_cor_2
                xi = root_grp.createVariable('mwave_t02', 'f8', ('t_ex','pix_ex'))
                xi[:] = mwave_t02
                xi = root_grp.createVariable('mwave_dir', 'f8', ('t_ex','pix_ex'))
                xi[:] = mwave_dir
                xi = root_grp.createVariable('swh_karin', 'f8', ('t_ex','pix_ex'))
                xi[:] = swh_karin
                xi = root_grp.createVariable('swh_nadir', 'f8', ('t_ex','pix_ex'))
                xi[:] = swh_nadir
                xi = root_grp.createVariable('swh_model', 'f8', ('t_ex','pix_ex'))
                xi[:] = swh_model
                xi = root_grp.createVariable('distance_to_coast', 'f8', ('t_ex','pix_ex'))
                xi[:] = distance_to_coast
                xi = root_grp.createVariable('heading_to_coast', 'f8', ('t_ex','pix_ex'))
                xi[:] = heading_to_coast
                xi = root_grp.createVariable('depth_or_elevation', 'f8', ('t_ex','pix_ex'))
                xi[:] = depth_or_elevation

                xi = root_grp.createVariable('utc_us', 'f8', ('t_us',))
                xi[:] = utc_us
                xi = root_grp.createVariable('lat_us', 'f8', ('t_us','pix_us'))
                xi[:] = lat_us
                xi = root_grp.createVariable('lon_us', 'f8', ('t_us','pix_us'))
                xi[:] = lon180_us
                xi = root_grp.createVariable('ssha_karin_2_cor_us', 'f8', ('t_us','pix_us'))
                xi[:] = ssha_karin_2_cor_us
                xi = root_grp.createVariable('dac_us', 'f8', ('t_us','pix_us'))
                xi[:] = dac_us
                xi = root_grp.createVariable('tO_us', 'f8', ('t_us','pix_us'))
                xi[:] = tO_us
                xi = root_grp.createVariable('mss_us', 'f8', ('t_us','pix_us'))
                xi[:] = mss_us
                xi = root_grp.createVariable('int_tide_us', 'f8', ('t_us','pix_us'))
                xi[:] = int_tide_us
                xi = root_grp.createVariable('tP_us', 'f8', ('t_us','pix_us'))
                xi[:] = tP_us
                xi = root_grp.createVariable('tE_us', 'f8', ('t_us','pix_us'))
                xi[:] = tE_us
                xi = root_grp.createVariable('height_cor_xover_us', 'f8', ('t_us','pix_us'))
                xi[:] = height_cor_xover_us
                xi = root_grp.createVariable('ssb_cor_2_us', 'f8', ('t_us','pix_us'))
                xi[:] = ssb_cor_2_us


