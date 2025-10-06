import os
import GPUtil
import json

import os

def get_free_gpus(num_gpus_needed):
    available_gpus = GPUtil.getAvailable(order='memory',
                                     limit=num_gpus_needed,
                                     maxLoad=1.0,
                                     includeNan=True,
                                     maxMemory=1.0)
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))

free_gpus = get_free_gpus(num_gpus)
print(free_gpus)
if free_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))

else:
    raise RuntimeError("No free GPUs available!")

DEBUG_MODE = False
if DEBUG_MODE:
    rank = int(os.environ.get("RANK", 0))
    import debugpy

    debugpy.listen(address = ('0.0.0.0', 5678 + rank))
    if rank == 0:
        debugpy.wait_for_client() 
    breakpoint()

with open("./config/Multi_rl.json", 'r') as f:
    args = json.load(f)


import random
import numpy as np
import torch


print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))


import torch
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", torch.cuda.get_device_name(0))


random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

from alg.Multi_RL import Multi2

agent = Multi2(args)
agent.update()
