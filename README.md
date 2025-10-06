# Multi<sup>2</sup> LLM-Agent
Multi<sup>2</sup>: A Multi-agent LLM Framework for Hierarchical Multi-turn Decision-making with Offline Reinforcement Learning

### Anaconda Installation
1. Install prerequisites (before installing Anaconda)
```
sudo apt-get update
sudo apt-get upgrade   
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
2. Download the [Anaconda installer(Linux version)](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)
3. Install Anaconda
```
sudo apt-get install python3 python python3-pip python-pip git
bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
```
### Environment Setup
1. Benchmark dependencies
Follow the instructions in the [ScienceWorld](https://github.com/allenai/ScienceWorld) and [AlfWorld](https://github.com/alfworld) to install the required environments.
2. Clone this repository
```
git clone https://github.com/park-sangeun/Multi-2-LLM-Agent.git
```
3. Create and activate a virtual environment
```
conda create -n Multi python=3.10 -y
conda activate Multi
pip install -r requirements.txt
```

### Model Training
1. Format Tuning (Supervised Fine-Tuning)
- Set the configuration and base model in ```./config/Multi_SFT.json```.
- Set the checkpoint paths in ```./alg/Multi_SFT.py```.
```
self.high_checkpoint_dir = (
    f"{args['check_path']}/{args['benchmark']}/"
    f"{args['alg_name']}/{args['model_name']}/high/{timestamp}"
)

self.low_checkpoint_dir = (
    f"{args['check_path']}/{args['benchmark']}/"
    f"{args['alg_name']}/{args['model_name']}/low/{timestamp}"
)
```
* If you want to train the model on the ALFWorld benchmark, run
```
python Data_collect.py
```
- Then start training
```
python train_Multi_SFT.py
```
2. Offline Reinforcement Learning for Multi<sup>2</sup>
- Set the load paths in ```./alg/Multi_RL.py```.
```
high_path = f"{args['check_path']}/{args['benchmark']}/glider_bc/{args['model_name']}/high/"
low_path = f"{args['check_path']}/{args['benchmark']}/glider_bc/{args['model_name']}/low/"
```
- Set the checkpoint paths in ```./alg/Multi_RL.py```.
```
self.high_checkpoint_dir = (
    f"{args['check_path']}/{args['benchmark']}/"
    f"{args['alg_name']}/{args['model_name']}/NoSFT/high/{timestamp}"
)
self.low_checkpoint_dir = (
    f"{args['check_path']}/{args['benchmark']}/"
    f"{args['alg_name']}/{args['model_name']}/NoSFT/low/{timestamp}"
)
self.critic_checkpoint_dir = (
    f"{args['check_path']}/{args['benchmark']}/"
    f"{args['alg_name']}/{args['model_name']}/NoSFT/critic/{timestamp}"
)
```
- Set the configuration and base model in ```./config/Multi_rl.json```.
- Run the training script
```
python train_Multi_RL.py
```

### Evaluation
- Set **self.high_checkpoint_dir** and **self.low_checkpoint_dir** (the model path) in ```./alg/eval_RL.py``` (to point to the trained model)
- Then run the evaluation
```
python eval_Multi.py
```
