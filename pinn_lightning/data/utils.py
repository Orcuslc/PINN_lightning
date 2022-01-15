import numpy as np
import torch

def tensor_grid(x):
        """build tensor grid for multiple parameters
        
        Arguments:
                x {tuple or list of np.array} -- parameters
        
        Returns:
                grid {np.ndarray} -- tensor grids

        Example:
                >>> tensor_grid(([1, 2], [3, 4], [5, 6, 7]))
                >>> np.array([[1, 3, 5],
                                        [1, 3, 6],
                                        [1, 3, 7],
                                        [1, 4, 5],
                                        [1, 4, 6],
                                        [1, 4, 7],
                                        [2, 3, 5],
                                        [2, 3, 6],
                                        [2, 3, 7],
                                        [2, 4, 5],
                                        [2, 4, 6],
                                        [2, 4, 7]])
        """
        return np.vstack(np.meshgrid(*x, indexing = 'ij')).reshape((len(x), -1)).T


def tensors_from_numpy(*arrays):
    return list(map(lambda x: torch.from_numpy(x), arrays))