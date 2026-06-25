import pyresample as pr
import numpy as np

def rechunk(x, chunks):
    needed_chunks = {d:chunks[d] for d in x.dims if d in chunks}
    return x.chunk(needed_chunks)

def _resample_slice( M, orig_grid=None, new_grid=None, resample_type='pr_gauss', 
                     radius_of_influence=120e3, gauss_sigma=None, weight_funcs=None, 
                     **kwargs ):
    if len(M) > 1:
        if resample_type == 'pr_gauss':
            if gauss_sigma is None: gauss_sigma = radius_of_influence/2
            new_field = pr.kd_tree.resample_gauss( orig_grid, M, new_grid, radius_of_influence=radius_of_influence, sigmas=gauss_sigma, **kwargs)
        elif resample_type == 'pr_nearest':
            new_field = pr.kd_tree.resample_nearest( orig_grid, M, new_grid, radius_of_influence=radius_of_influence, **kwargs)
        elif resample_type == 'pr_custom':
            if weight_funcs is None: raise Exception("resample_slice error: weight function(s) (weights_funcs) must be specified for a custom resample")
            new_field = pr.kd_tree.resample_nearest( orig_grid, M, new_grid, radius_of_influence=radius_of_influence, weight_funcs=weight_funcs, **kwargs)
        else:
            raise Exception(f"resample_slice error: {resample_type} does not match a compatible resample_type [pr_gauus, pr_nearest, pr_custom]")
    else:
        new_field = np.full(new_grid.shape, M[0])
    return new_field