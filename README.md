# Multi<sup>2</sup> LLM-Agent
Multi<sup>2</sup>: A Multi-agent LLM Framework for Hierarchical Multi-turn Decision-making with Offline Reinforcement Learning

### Anaconda Download
1. Before Anaconda Download
```
sudo apt-get update
sudo apt-get upgrade   
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
2. [Anaconda installer(Linux version)](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh) Download
3. After Anaconda Download
```
sudo apt-get install python3 python python3-pip python-pip git
bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
```
### Environment Setting
1. Benchmark Setting
Follow the instructions in the [ScienceWorld](https://github.com/allenai/ScienceWorld) and [AlfWorld](https://github.com/alfworld) to install.
2. Download Files
```
git clone https://github.com/park-sangeun/Multi-2-LLM-Agent.git
```
3. Create Virtual Environment
```
conda create -n Multi python=3.10 -y
conda activate Multi
pip install -r requirements.txt
```

### Train Model
1. Format Tuning with Supervised Fine-tuning
- Set configuration and base model in ```./config/Multi_SFT.json```.
- Set the model path that you want in ```./alg/Multi_SFT.py```.
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
* If you want to train the model on the ALFWorld benchmark, run the following command.
```
python Data_collect.py
```
- Run
```
python train_Multi_SFT.py
```
2. Offline Reinforcement Learning for Multi<sup>2</sup>
- Set the load path in ```./alg/Multi_RL.py```.
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
- Set configuration and base model in ```./config/Multi_rl.json```.
- Run
```
python train_Multi_RL.py
```

### Evaluation
- Set **self.high_checkpoint_dir** and **self.low_checkpoint_dir** (the model path) in ```./alg/eval_RL.py```.
```
python eval_Multi.py
```
