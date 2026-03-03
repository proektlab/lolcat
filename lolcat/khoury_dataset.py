import os
from typing import Literal, Union, Sequence, get_args

import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data
from lolcat import InMemoryDataset
from lolcat.utils import compute_isi_distribtuion

cell_types = ['red', 'non']  # red = SOM+, non = SOM-
cell_type_names = ['SOM+', 'SOM-']
KhouryExpType = Literal['pass', 'spont']


class KhouryDataset(InMemoryDataset):
    """Load Khoury et al. SOM interneuron +/- data (from mat files)"""
    fr = 30  # frames/s
    
    def __init__(self, subj_and_dates: list[str], root="F:\\khoury_som_data",
                 exp_types: Union[KhouryExpType, Sequence[KhouryExpType]] = ('pass', 'spont'),
                 region: Literal['ac', 'ppc'] = 'ppc', concat=False,
                 transform=None, force_process=False, trial_length=3.):
        self.trial_length = trial_length
        self.window_size = int(trial_length * self.fr)
        self.subj_and_dates = subj_and_dates
        self.region = region
        self.exp_types = (exp_types,) if not isinstance(exp_types, Sequence) else exp_types
        self.concat = concat  # whether to concatenate passive and spontaneous together

        name = '_'.join([','.join(subj_and_dates), region, ','.join(self.exp_types), f'{self.window_size}samp'])
                
        super().__init__(root, name, transform, force_process)

    def process(self):
        # load requested data from MAT files and get ISI distributions
        cell_list: list[Data] = []
        cell_id_list: list[str] = []
        type_counts = np.zeros(len(cell_types), dtype=int)

        for subj in self.subj_and_dates:
            data_file = os.path.join(self.root, f'{subj} {self.region}.mat')
            mat_vars = scipy.io.loadmat(data_file)

            for y, cell_type in enumerate(cell_types):
                subj_data: list[np.ndarray] = []
                exp_type_inds = []

                for exp_type in self.exp_types:
                    event_mat = mat_vars[f'deconv{cell_type}{exp_type}']  # cell x time
                    # split into cells x trials x time
                    n_trials = event_mat.shape[1] // self.window_size
                    time_to_use = n_trials * self.window_size
                    event_mat = np.reshape(event_mat[:, :time_to_use], (event_mat.shape[0], n_trials, self.window_size), order='F')
                    subj_data.append(event_mat)
                    exp_type_inds.append(get_args(KhouryExpType).index(exp_type))
                
                if self.concat:
                    subj_data = [np.concatenate(subj_data, axis=1)]
                    exp_type_inds = [len(get_args(KhouryExpType))]
                
                for event_mat, exp_type_ind in zip(subj_data, exp_type_inds):
                    # process into ISI distribution
                    type_counts[y] += event_mat.shape[0]
                    isi = compute_isi_distribtuion(event_mat)
                    for cell_ind, cell_isi in enumerate(isi):
                        cell_id = subj + f' cell{cell_ind}'
                        cell_data = Data(
                            x=torch.FloatTensor(cell_isi),
                            y=torch.tensor(y),
                            exp_type=torch.tensor(exp_type_ind),
                            subj_date=subj,
                            cell_id=cell_id
                        )
                        cell_list.append(cell_data)
                        cell_id_list.append(cell_id)
        
        cell_ids = np.array(cell_id_list)
        return {'data_list': cell_list, 'cell_ids': cell_ids, 'type_proportions': type_counts / type_counts.sum()}
