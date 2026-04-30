from typing import List
import numpy as np

import os
import sys
import warnings
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.insert(0, import_dir)
from catch import CATCH


warnings.filterwarnings('ignore')

def diffusion_condensation(X: np.ndarray,
                           num_workers: int = 1,
                           random_seed: int = 0) -> List[np.ndarray]:
    '''
    `X` : [N, D] feature matrix,
        where N := number of feature vectors
              D := number of features
    '''
    try:
        dc_op = CATCH(n_pca=None,
                      t='auto',
                      random_state=random_seed,
                      n_jobs=num_workers)

        dc_op.fit_transform(X)
        return dc_op.Xs

    except:
        print('Diffusion condensation fails.')
        return None

