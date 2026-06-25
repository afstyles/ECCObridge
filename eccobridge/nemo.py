import glob
import xnemogcm
import os
import xarray as xr
from xarray import apply_ufunc
import pyresample as pr
import ecco_v4_py as ecco
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from .universal import rechunk, _resample_slice

def _dims_from_nemo_field( nemo_field ):
    if 'x_f' in nemo_field.dims: 
        nemo_xdim = 'x_f'
        ecco_idim = 'i_g'
    else:
        nemo_xdim = 'x_c'
        ecco_idim = 'i'
    
    if 'y_f' in nemo_field.dims:
        nemo_ydim = 'y_f'
        ecco_jdim = 'j_g'
    else:
        nemo_ydim = 'y_c'
        ecco_jdim = 'j'
    
    return nemo_xdim, nemo_ydim, ecco_idim, ecco_jdim

def _dims_from_ecco_field( ecco_field ):
    if 'i_g' in ecco_field.dims: 
        nemo_xdim = 'x_f'
        ecco_idim = 'i_g'
    else:
        nemo_xdim = 'x_c'
        ecco_idim = 'i'
    
    if 'j_g' in ecco_field.dims:
        nemo_ydim = 'y_f'
        ecco_jdim = 'j_g'
    else:
        nemo_ydim = 'y_c'
        ecco_jdim = 'j'
    
    return nemo_xdim, nemo_ydim, ecco_idim, ecco_jdim
    
def _get_nemo_lonlat( ds_nemo_grid, nemo_xdim, nemo_ydim ):
    if (nemo_xdim == 'x_c') and (nemo_ydim == 'y_c'): 
        nemo_lon, nemo_lat = ds_nemo_grid['glamt'], ds_nemo_grid['gphit']  #T-point case
    elif (nemo_xdim == 'x_f') and (nemo_ydim == 'y_c'): 
        nemo_lon, nemo_lat = ds_nemo_grid['glamu'], ds_nemo_grid['gphiu']  #U-point case
    elif (nemo_xdim == 'x_c') and (nemo_ydim == 'y_f'): 
        nemo_lon, nemo_lat = ds_nemo_grid['glamv'], ds_nemo_grid['gphiv']  #V-point case
    else:
        raise Exception(f"ERROR: Dimensions do not match known types [x_c, x_f, y_c, y_f]")

    return nemo_lon, nemo_lat

def _get_ecco_lonlat( ds_ecco_grid, ecco_idim, ecco_jdim):
    ecco_xgcm_grid = ecco.get_llc_grid(ds_ecco_grid)

    if (ecco_idim == 'i'  ) and (ecco_jdim == 'j'  ):  #T-point case
        ecco_lon, ecco_lat = ds_ecco_grid['XC'], ds_ecco_grid['YC']
    elif (ecco_idim == 'i_g') and (ecco_jdim == 'j'  ):  #U-point case
        ecco_lon, ecco_lat = ecco_xgcm_grid.interp(ds_ecco_grid['XC'], 'X'), ecco_xgcm_grid.interp(ds_ecco_grid['YC'], 'X')
    elif (ecco_idim == 'i'  ) and (ecco_jdim == 'j_g'):  #V-point case
        ecco_lon, ecco_lat = ecco_xgcm_grid.interp(ds_ecco_grid['XC'],'Y')  , ecco_xgcm_grid.interp(ds_ecco_grid['YC'],'Y')
    else:
        raise Exception(f"ERROR: Dimensions do not match known types [i, i_g, j, j_g]")
    
    return ecco_lon, ecco_lat



def resample_nemo( nemo_field, ecco_field, ds_nemo_grid, ds_ecco_grid, resample_type='pr_gauss',
                   radius_of_influence=120e3, gauss_sigma=None, weight_funcs=None,
                   **kwargs):

    if   (nemo_field is not None) and (ecco_field is None):
        nemo2ecco = True
        print("Resampling NEMO data onto ECCO grid")
    elif (nemo_field is None) and (ecco_field is not None):
        nemo2ecco = False
        print("Resampling ECCO data onto NEMO grid")
    else:
        raise Exception("ERROR: Incorrect inputs for nemo_field and/or ecco_field")


    # Identify the relevant dimensions on the staggered C-grid >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if nemo2ecco:
        nemo_xdim, nemo_ydim, ecco_idim, ecco_jdim = _dims_from_nemo_field(nemo_field)
    else:
        nemo_xdim, nemo_ydim, ecco_idim, ecco_jdim = _dims_from_ecco_field(ecco_field)
    
    # Load the longitude and latitude dimensions from the grid datasets
    nemo_lon, nemo_lat = _get_nemo_lonlat( ds_nemo_grid, nemo_xdim, nemo_ydim )
    ecco_lon, ecco_lat = _get_ecco_lonlat( ds_ecco_grid,   ecco_idim, ecco_jdim )

    # Count the number of x and y points on the ORCA grid
    nemo_nx = len(ds_nemo_grid[nemo_xdim])
    nemo_ny = len(ds_nemo_grid[nemo_ydim])

    # Count the number of i, j, and tile points on the ECCO LLC grid
    ecco_ni = len(ds_ecco_grid[ecco_idim])
    ecco_nj = len(ds_ecco_grid[ecco_jdim])
    ecco_ntile = len(ds_ecco_grid['tile'])

    print("NEMO: >>>>>>>>")
    print(f"nx: {nemo_nx} [dim: {nemo_xdim}]")
    print(f"ny: {nemo_ny} [dim: {nemo_ydim}]")
    print("ECCO >>>>>>>>>")
    print(f"ni: {ecco_ni} [dim: {ecco_idim}]")
    print(f"ni: {ecco_nj} [dim: {ecco_jdim}]")
    print(f"ntiles: {ecco_ntile} [dim: tile]")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Sort NEMO dimensions into horizontal dimensions (y, x) and non-horizontal dimensions
    nemo_xydims = [nemo_ydim, nemo_xdim]
    if nemo2ecco == True: nemo_nonxydims = [d for d in nemo_field.dims if d not in nemo_xydims]

    # Sort ECCO dimensions into horizontal dimensions (tile, j, i)
    ecco_xydims    = ['tile', ecco_jdim, ecco_idim]
    if nemo2ecco == False: ecco_nonxydims = [d for d in ecco_field.dims if d not in ecco_xydims]
    
    # Stack the ecco fields as the LLC is not a regular grid
    ecco_lon_stack = ecco_lon.stack(xy=ecco_xydims)
    ecco_lat_stack = ecco_lat.stack(xy=ecco_xydims)
    xy_index = ecco_lon_stack.xy
    nxy = len(xy_index)

    # Stack ECCO field as well if regridding ECCO data onto NEMO grid
    if nemo2ecco == False: 
        ecco_field_stack = ecco_field.stack(xy=ecco_xydims, nonxy=ecco_nonxydims).chunk({'xy':-1})

    # Define the regular NEMO grid for pyresample
    nemo_grid = pr.geometry.GridDefinition( nemo_lon.values , nemo_lat.values )

    # Define the irregular ECCO grid for pyresample
    ecco_grid = pr.geometry.SwathDefinition( ecco_lon_stack.values , ecco_lat_stack.values )

    # NEMO -> ECCO
    if nemo2ecco == True:
        return apply_ufunc(_resample_slice, nemo_field, input_core_dims=[nemo_xydims], output_core_dims=[['xy']], 
                       output_sizes={'xy': nxy},
                       vectorize=True, dask='parallelized',
                       kwargs={'orig_grid':nemo_grid, 
                               'new_grid':ecco_grid, 
                               'radius_of_influence':radius_of_influence, 
                               'resample_type': resample_type,
                               'gauss_sigma': gauss_sigma,
                               'weight_funcs': weight_funcs,
                                **kwargs}).assign_coords({'xy':xy_index}).unstack()
    
    # ECCO -> NEMO
    else:
        return apply_ufunc(_resample_slice, ecco_field_stack, input_core_dims=[['xy']], output_core_dims=[nemo_xydims], 
                output_sizes={nemo_ydim: nemo_ny, nemo_xdim: nemo_nx},
                vectorize=True, dask='parallelized',
                kwargs={'orig_grid':ecco_grid, 
                        'new_grid':nemo_grid, 
                        'radius_of_influence':radius_of_influence, 
                        'resample_type': resample_type,
                        'gauss_sigma': gauss_sigma,
                        'weight_funcs': weight_funcs,
                         **kwargs}).unstack()



