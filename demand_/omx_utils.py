"""
omx_utils.py
Lightweight OMX read/write using h5py (avoids openmatrix/tables binary incompatibility).
OMX spec: HDF5 with /data/{matrix_name} datasets, /lookup/{mapping_name} dataset.
"""
import numpy as np
import h5py


def write_omx(filepath, matrices, zone_mapping=None, shape=None):
    """
    Write an OMX file.

    Parameters
    ----------
    filepath : str
        Output path.
    matrices : dict {name: 2D np.ndarray}
        Matrix name -> numpy array.
    zone_mapping : dict {str_zone_id: int_index} or None
        Zone ID string -> 0-based integer index mapping.
    shape : tuple (n, n) or None
        If None, inferred from first matrix.
    """
    if shape is None:
        first = next(iter(matrices.values()))
        shape = first.shape

    with h5py.File(filepath, 'w') as f:
        # OMX attributes
        f.attrs['OMX_VERSION'] = '0.2'
        f.attrs['SHAPE'] = np.array(shape, dtype=np.int32)

        # Create data group
        data_grp = f.create_group('data')
        for name, mat in matrices.items():
            arr = np.array(mat, dtype=np.float32)
            ds = data_grp.create_dataset(name, data=arr, compression='lzf')
            ds.attrs['_FillValue'] = np.float32(0.0)

        # Create lookup group
        lookup_grp = f.create_group('lookup')
        if zone_mapping is not None:
            n = shape[0]
            # Store as array of strings, index = position
            ids_by_idx = [''] * n
            for zone_str, idx in zone_mapping.items():
                if 0 <= idx < n:
                    ids_by_idx[idx] = str(zone_str)
            arr = np.array(ids_by_idx, dtype='S20')
            lookup_grp.create_dataset('SA2_CODE', data=arr)


def read_omx_matrix(filepath, matrix_name):
    """Read a single matrix from an OMX file."""
    with h5py.File(filepath, 'r') as f:
        return np.array(f['data'][matrix_name], dtype=np.float32)


def list_omx_matrices(filepath):
    """List matrix names in an OMX file."""
    with h5py.File(filepath, 'r') as f:
        return list(f['data'].keys())
