import torch
import random
import numpy as np



def set_seed(global_seed):
    torch.manual_seed(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    # np random number generators???
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    # torch.use_deterministic_algorithms(True)

    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)

