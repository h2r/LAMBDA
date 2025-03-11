import h5py

def print_hdf5_group_structure(group, prefix=''):
    """
    Recursively prints the structure of an HDF5 group (which could be the file itself).
    
    Parameters:
    - group: An instance of h5py.File or h5py.Group.
    - prefix: A string prefix used to indicate the current level in the hierarchy.
    """
    for key in group.keys():
        item = group[key]
        print(prefix + key)
        if isinstance(item, h5py.Group):  # If the current item is a group, recurse into it
            print_hdf5_group_structure(item, prefix=prefix + '  ')
        elif isinstance(item, h5py.Dataset):  # If it's a dataset, print its shape
            print(prefix + f'  [Dataset] shape: {item.shape}, dtype: {item.dtype}')

# Open the HDF5 file
hdf5_path = '/users/ajaafar/data/shared/lanmp/lanmp_dataset.hdf5'
with h5py.File(hdf5_path, 'r') as hdf:
    print_hdf5_group_structure(hdf)
