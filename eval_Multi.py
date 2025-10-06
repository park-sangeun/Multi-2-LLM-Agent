import os
import GPUtil
import json


def get_free_gpus(num_gpus_needed):
    available_gpus = GPUtil.getAvailable(order='memory',
                                     limit=num_gpus_needed,
                                     maxLoad=1.0,
                                     includeNan=True,
                                     maxMemory=1.0)
    return available_gpus

num_gpus = int(os.environ.get('WORLD_SIZE', 1))

free_gpus = get_free_gpus(num_gpus)
if free_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, free_gpus))

else:
    raise RuntimeError("No free GPUs available!")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEBUG_MODE = False
if DEBUG_MODE:
    rank = int(os.environ.get("RANK", 0))
    import debugpy

    debugpy.listen(address = ('0.0.0.0', 5678 + rank))
    if rank == 0:
        debugpy.wait_for_client() 
    breakpoint()

with open("./config/eval_RL.json", 'r') as f:
    args = json.load(f)
print(args)


from alg.eval_RL import EvalAgent
import random
import numpy as np
import torch

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

eval_agent = EvalAgent(args)
eval_agent.evaluate_online(15, "test")
