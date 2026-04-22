import scipy.io as sio
import numpy as np
import os

def extract_oxford_dataset(mat_filepath):
    """
    Extracts the Oxford Battery Degradation Dataset from a .mat file.
     Returns a Python dictionary structured by Cells, Cycles, Phases, and Variables.
    
    Structure:
    {
        'Cell1': {
            'cyc0000': {
                'C1ch': {'t': array, 'v': array, 'q': array, 'T': array},
                'C1dc': {...},
                'OCVch': {...},
                'OCVdc': {...}
            },
            ...
        },
        ...
    }
    """
    if not os.path.exists(mat_filepath):
        raise FileNotFoundError(f"The file {mat_filepath} does not exist.")

    print(f"Loading {mat_filepath}...")
    mat = sio.loadmat(mat_filepath)
    
    dataset = {}
    
    # Identify Cell keys
    cell_keys = [k for k in mat.keys() if k.startswith('Cell')]
    
    for cell_name in cell_keys:
        print(f"Extracting {cell_name}...")
        cell_data = mat[cell_name][0, 0]
        dataset[cell_name] = {}
        
        # Identify cycle names (cyc0000, cyc0100, etc.)
        cycle_names = [name for name in cell_data.dtype.names if name.startswith('cyc')]
        
        for cyc_name in cycle_names:
            cyc_struct = cell_data[cyc_name][0, 0]
            dataset[cell_name][cyc_name] = {}
            
            # Identify phases (C1ch, C1dc, OCVch, OCVdc)
            phases = [p for p in cyc_struct.dtype.names if not p.startswith('__')]
            
            for phase in phases:
                phase_struct = cyc_struct[phase][0, 0]
                dataset[cell_name][cyc_name][phase] = {}
                
                # Identify variables (t, v, q, T)
                if phase_struct.dtype is not None and phase_struct.dtype.names is not None:
                    for var in phase_struct.dtype.names:
                        # Flatten and convert to numpy array
                        data = phase_struct[var].flatten()
                        dataset[cell_name][cyc_name][phase][var] = data
                
    return dataset

if __name__ == "__main__":
    # Example usage:
    mat_file = "Oxford_Battery_Degradation_Dataset_1.mat"
    if os.path.exists(mat_file):
        try:
            data = extract_oxford_dataset(mat_file)
            print("Extraction successful!")
            print(f"Number of cells: {len(data.keys())}")
            print(data)
            print(f"Sample data from Cell1, cyc0000, C1ch: {data['Cell1']['cyc0000']['C1ch'].keys()}")
        except Exception as e:
            print(f"Error during extraction: {e}")
    else:
        print(f"File {mat_file} not found.")
