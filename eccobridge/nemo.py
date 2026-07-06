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
import copy

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
                   periodic_x=True, periodic_y=False, padlength_x=1, padlength_y=1,
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

    # If there are NaNs in the longitude or latitude field (there really shouldn't be)
    if bool(nemo_lon.isnull().max().values) or bool(nemo_lat.isnull().max().values):
        print('WARNING masked values in the NEMO longitude and/or latitude array')
        print('NEMO grid will have to be treated as irregular which may affect performance')
        nemo_swath = True
    else:
        nemo_swath = False 

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

    if nemo_swath == True:
        nemo_lon_stack = nemo_lon.stack(nemo_xy=nemo_xydims)
        nemo_lat_stack = nemo_lat.stack(nemo_xy=nemo_xydims)
        if nemo2ecco == True: nemo_field_stack = nemo_field.stack(nemo_xy=nemo_xydims, nemo_nonxy=nemo_nonxydims)
        nemo_xy_index = nemo_lon_stack.nemo_xy
        nemo_nxy = len(nemo_xy_index)

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

    
    if nemo_swath == False:
        # Define the regular NEMO grid for pyresample
        nemo_grid = pr.geometry.GridDefinition( nemo_lon.values , nemo_lat.values )
    else:
        nemo_grid = pr.geometry.SwathDefinition( nemo_lon_stack.values , nemo_lat_stack.values )

    # Define the irregular ECCO grid for pyresample
    ecco_grid = pr.geometry.SwathDefinition( ecco_lon_stack.values , ecco_lat_stack.values )

    # NEMO grid -> ECCO swath
    if (nemo2ecco == True) and (nemo_swath == False):
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
    
    # NEMO swath -> ECCO swath
    elif (nemo2ecco == True) and (nemo_swath == True):
        return apply_ufunc(_resample_slice, nemo_field_stack, input_core_dims=[['nemo_xy']], output_core_dims=[['xy']], 
                        output_sizes={'xy': nxy},
                        vectorize=True, dask='parallelized',
                        kwargs={'orig_grid':nemo_grid, 
                                'new_grid':ecco_grid, 
                                'radius_of_influence':radius_of_influence, 
                                'resample_type': resample_type,
                                'gauss_sigma': gauss_sigma,
                                'weight_funcs': weight_funcs,
                                **kwargs}).assign_coords({'xy':xy_index}).unstack()
    
    # ECCO swath -> NEMO grid
    elif (nemo2ecco == False) and (nemo_swath == False):
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
    
    # ECCO swath -> NEMO swath
    else:
        return apply_ufunc(_resample_slice, ecco_field_stack, input_core_dims=[['xy']], output_core_dims=[['nemo_xy']], 
                output_sizes={'nemo_nxy': nemo_nxy},
                vectorize=True, dask='parallelized',
                kwargs={'orig_grid':ecco_grid, 
                        'new_grid':nemo_grid, 
                        'radius_of_influence':radius_of_influence, 
                        'resample_type': resample_type,
                        'gauss_sigma': gauss_sigma,
                        'weight_funcs': weight_funcs,
                         **kwargs}).unstack()
    
def _interpolate_masked_nemofield( nemo_field, periodic_x=True, periodic_y=False, padlength_x=1, padlength_y=1 ):

    # Get the dimension names for the nemo field
    nemo_xdim, nemo_ydim, _, _ = _dims_from_nemo_field( nemo_field )

    if padlength_x is None: padlength_x = len(nemo_xdim)
    if padlength_y is None: padlength_y = len(nemo_ydim)

    # If periodic, pad the fields for appropriate interpolation
    if periodic_x == True:
        xcoord0 = nemo_field[nemo_xdim]
        pad_opts = {nemo_xdim:padlength_x, 'mode':'wrap'}
        nemo_field = nemo_field.pad(**pad_opts).drop(nemo_xdim).chunk({nemo_xdim:-1})
    
    if periodic_y == True:
        ycoord0 = nemo_field[nemo_ydim]
        pad_opts = {nemo_ydim:padlength_y, 'mode':'wrap'}
        nemo_field = nemo_field.pad(**pad_opts).drop(nemo_ydim).chunk({nemo_ydim:-1})

    nemo_field = nemo_field.interpolate_na(dim=nemo_xdim)
    nemo_field = nemo_field.interpolate_na(dim=nemo_ydim)

    # If periodic, remove any padding and restore original coordinates
    if periodic_x == True:
        nemo_field = nemo_field.isel(**{nemo_xdim:slice(padlength_x,-padlength_x)})
        nemo_field = nemo_field.assign_coords({nemo_xdim:xcoord0})

    if periodic_y == True:
        nemo_field = nemo_field.isel(**{nemo_ydim:slice(padlength_y,-padlength_y)})
        nemo_field = nemo_field.assign_coords({nemo_ydim:ycoord0})

    return nemo_field







