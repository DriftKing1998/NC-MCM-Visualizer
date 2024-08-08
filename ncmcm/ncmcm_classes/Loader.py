import numpy as np
import os, mat73


class Loader:
    """
    Reads in the data from the all files corresponding to the selected dataset.
    It stores all values into numpy arrays.

    Parameters:
        - data_set_no: int, required
            Defines which CSV files will be read.

        - path: str, optional
            Path to the matlab-file
    """
    def __init__(self,
                 data_set_no,
                 path=None):
        """
        Reads in the data from the all files corresponding to the selected dataset.
        It stores all values into numpy arrays.

        Parameters:
            
            - data_set_no: int, required
                Defines which CSV files will be read.

            - path: str, optional
                Path to the matlab-file
        """

        if path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, '..', 'data', 'datasets', 'NoStim_Data.mat')

        self.data_set_no = data_set_no
        data_dict = mat73.loadmat(path)
        data = data_dict['NoStim_Data']
        deltaFOverF_bc = data['deltaFOverF_bc'][self.data_set_no]
        derivatives = data['derivs'][self.data_set_no]
        NeuronNames = data['NeuronNames'][self.data_set_no]
        fps = float(data['fps'][self.data_set_no])
        States = data['States'][self.data_set_no]

        self.B = np.sum([n * States[s] for n, s in enumerate(States)], axis=0).astype(
            int)  # making a single states array in which each number corresponds to a behaviour
        self.states = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        self.data = self.neuron_traces, self.B, self.neuron_names, self.states, self.fps

