import os
import deepspeed
import torch
from transformers import AutoTokenizer
from util.model import Policy, HighPolicy, LowPolicy
from util.replay_buffer import HierarchyDataset, batch_traj_process
from alg.bc import Agent
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from prompt.inst import high_prompt, low_prompt

from datetime import datetime
import wandb
from torch.utils.data import Subset, DataLoader


class Multi2:
    def __init__(self, args):
        self.args = args
        self.high_policy = HighPolicy(args)
        self.high_policy.train()
        if hasattr(self.high_policy, "base"):
            self.high_policy.base.train()
        self.low_policy = LowPolicy(args)
        self.low_policy.train()
        if hasattr(self.low_policy, "base"):
            self.low_policy.base.train()

    
        self.high_engine, *_ = deepspeed.initialize(
            model=self.high_policy,
            model_parameters=[{"params": [p for p in self.high_policy.base.parameters() if p.requires_grad],
                               "lr": args["lr_high"]}],
            config=args["ds_config"]
        )
        
        self.low_engine, *_ = deepspeed.initialize(
            model=self.low_policy,
            model_parameters=[{"params": [p for p in self.low_policy.base.parameters() if p.requires_grad],
                               "lr": args["lr_low"]}],
            config=args["ds_config"]
        )
        
        self.buffer = HierarchyDataset(args)

        if self.high_engine.global_rank == 0:
           
            log_dir_high = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/high"
            self.high_writer = SummaryWriter(log_dir=log_dir_high)

        if self.low_engine.global_rank == 0:
           
            log_dir_low = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/low"
            self.low_writer = SummaryWriter(log_dir=log_dir_low)

        self.high_global_step = torch.tensor(0, dtype=torch.int64).to(self.high_engine.device)
        self.low_global_step = torch.tensor(0, dtype=torch.int64).to(self.low_engine.device)

    
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.high_checkpoint_dir = (
            f"{args['check_path']}/{args['benchmark']}/"
            f"{args['alg_name']}/{args['model_name']}/high/{timestamp}"
        )

        self.low_checkpoint_dir = (
            f"{args['check_path']}/{args['benchmark']}/"
            f"{args['alg_name']}/{args['model_name']}/low/{timestamp}"
        )

    def update_policy(self):
        batch_size_per_gpu = min(
            self.high_engine.train_micro_batch_size_per_gpu(),
            self.low_engine.train_micro_batch_size_per_gpu()
        )
        
        dataloader = DataLoader(
            self.buffer,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            collate_fn=HierarchyDataset.collate_fn
        )
        for epoch in range(self.args['epochs']):
            high_epoch_loss, low_epoch_loss = 0.0, 0.0
            for batch in dataloader:
                """
                Args:
                    batch:{
                        "task_description": [traj_nums,] or "subtask":[group_nums, ]
                        "obs":[traj_nums, steps+1],      or "obs": [group_nums, steps+1]
                        "subtask":[traj_nums, steps],    or "action": [group_nums, steps]
                        ...
                        }
                    pi(a|s)/action_token_len   
                """
                batch_high_tokens = batch_traj_process(batch['high']['task_description'],
                                       batch['high']['obs'],
                                       batch['high']['subtask'],
                                       self.high_policy.tokenizer).to(self.high_engine.device)

                batch_low_tokens = batch_traj_process(batch['low']['subtask'],
                                      batch['low']['obs'],
                                      batch['low']['action'],
                                      self.low_policy.tokenizer).to(self.low_engine.device)


                high_log_probs, high_masks = self.high_policy.get_log_prob(batch_high_tokens)
                high_valid_log_prob = Agent.extract_valid_action_probs(self, high_log_probs, high_masks,
                                                                    max(batch_high_tokens['action_end_mask'].sum(dim=1)))
                high_loss = -high_valid_log_prob.mean()

                self.high_engine.backward(high_loss)
                self.high_engine.step()

                low_log_probs, low_masks = self.low_policy.get_log_prob(batch_low_tokens)
                low_valid_log_prob = Agent.extract_valid_action_probs(self, low_log_probs, low_masks,
                                                                    max(batch_low_tokens['action_end_mask'].sum(dim=1)))
                low_loss = -low_valid_log_prob.mean()

                self.low_engine.backward(low_loss)
                self.low_engine.step()

                high_epoch_loss += high_loss.item()
                low_epoch_loss += low_loss.item()
                self.high_global_step += 1
                self.low_global_step += 1

                if self.high_engine.local_rank == 0: 
                    print(f"[HIGH] train; step:{self.high_global_step.item()}; loss:{high_loss.item()}")
                    self.high_writer.add_scalar('step_loss', high_loss.item(), self.high_global_step.item())
                   

                if self.low_engine.local_rank == 0: 
                    print(f"[LOW] train; step:{self.low_global_step.item()}; loss:{low_loss.item()}")
                    self.low_writer.add_scalar('step_loss', low_loss.item(), self.low_global_step.item())
                  

                
                if self.high_engine.local_rank == 0 and self.high_global_step.item() % self.args['eval_freq'] == 0 :
                    print(f"[HIGH] hierarcy-train; epoch {epoch}, loss:{high_epoch_loss}")
                    Agent.save_off_policy(self, self.high_engine, self.high_global_step.item(), self.high_checkpoint_dir) 
                    self.high_writer.add_scalar('epoch_loss', high_epoch_loss, epoch)

                    print(f"[LOW] hierarcy-train; epoch {epoch}, loss:{low_epoch_loss}")
                    Agent.save_off_policy(self, self.low_engine, self.high_global_step.item(), self.low_checkpoint_dir)  
                    self.low_writer.add_scalar('epoch_loss', low_epoch_loss, epoch)


            print(f"[HIGH] hierarcy-train; epoch {epoch}, loss:{high_epoch_loss}")
            Agent.save_off_policy(self, self.high_engine, self.high_global_step.item(), self.high_checkpoint_dir) 
            self.high_writer.add_scalar('epoch_loss', high_epoch_loss, epoch)

            print(f"[LOW] hierarcy-train; epoch {epoch}, loss:{low_epoch_loss}")
            Agent.save_off_policy(self, self.low_engine, self.high_global_step.item(), self.low_checkpoint_dir)  
            self.low_writer.add_scalar('epoch_loss', low_epoch_loss, epoch)
                    
