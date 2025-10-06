from util.model import Policy, Critic, HighPolicy, LowPolicy
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import HierarchyDataset, batch_traj_process
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import deepspeed
import torch
import copy
import pandas as pd
import random
import os
from prompt.inst import high_prompt, low_prompt

from util.extract import extract_action_done
from scienceworld import ScienceWorldEnv
from util.replay_buffer import OnlineDataset

            
class Multi2:
    def __init__(self, args):
        self.args = args
        self.critic = Critic(args)
     
        self.high_policy = HighPolicy(args)
        self.high_policy.train()
        if hasattr(self.high_policy, "base"):
            self.high_policy.base.train()
        self.low_policy = LowPolicy(args)
        self.low_policy.train()
        if hasattr(self.low_policy, "base"):
            self.low_policy.base.train()
        self.critic_engine, *_ = deepspeed.initialize(
            model=self.critic,
            model_parameters=[{"params": [p for p in self.critic.parameters() if p.requires_grad],
                            "lr": args["critic_lr"]}],
            config=args["ds_config"]
        )

        self.high_engine, *_ = deepspeed.initialize(
            model=self.high_policy,
            model_parameters=self.high_policy.base.parameters(),
            config=args["ds_config"]
        )

        self.low_engine, *_ = deepspeed.initialize(
            model=self.low_policy,
            model_parameters=self.low_policy.base.parameters(),
            config=args["ds_config"]
        )
        
        BC_AGENT.load_high_policy(self, high_path)
        BC_AGENT.load_low_policy(self, low_path)
        
        
        self.loss_fct = torch.nn.MSELoss()
        self.buffer = HierarchyDataset(args)


        if self.high_engine.global_rank == 0:
            log_dir_high = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline/high"
            self.high_writer = SummaryWriter(log_dir=log_dir_high)

        if self.low_engine.global_rank == 0:
            log_dir_low = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline/low"
            self.low_writer = SummaryWriter(log_dir=log_dir_low)
        
        if self.critic_engine.global_rank == 0:
            log_dir_critic = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline/critic"
            self.critic_writer = SummaryWriter(log_dir=log_dir_critic)

        self.high_engine.train()  
        self.high_engine.module.train()
        self.low_engine.train()
        self.low_engine.module.train()
        self.critic_engine.train()
        self.critic_engine.module.train()
        self.high_global_step = torch.tensor(0, dtype=torch.int64).to(self.high_engine.device)
        self.low_global_step = torch.tensor(0, dtype=torch.int64).to(self.low_engine.device)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    
    def update_policy(self, batch_data, level="high"):
        micro_batch_size = 4  
        total_size = len(batch_data['obs'])
        def expectile_loss(diff: torch.Tensor, expectile: float = 0.7):
            """
            diff: Q(s,a) - V(s) (stop grad on Q)
            expectile: 0.5이면 MSE, 0.9에 가까울수록 overestimation 쪽으로 기울어짐
            """
            weight = torch.where(diff > 0, expectile, 1 - expectile)
            return (weight * (diff ** 2)).mean()
        if level == "high":
            engine = self.high_engine
            bc_loss_total = 0.0

            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}

                batch_tokens = batch_traj_process(batch_slice['task_description'],
                                                batch_slice['obs'],
                                                batch_slice['subtask'],
                                                engine.tokenizer).to(engine.device)
                action_log_probs, action_masks = engine.get_log_prob(batch_tokens)
                valid_action_log_probs = BC_AGENT.extract_valid_action_probs(
                    self, action_log_probs, action_masks,
                    max(batch_tokens['action_end_mask'].sum(dim=1))
                )
                bc_loss = -valid_action_log_probs.mean()
                engine.backward(bc_loss)
                engine.step()

                bc_loss_total += bc_loss.item() * len(batch_slice['obs'])

            return bc_loss_total / total_size

        elif level == "low":
            engine = self.low_engine
            q_loss_total = 0.0
            v_loss_total = 0.0
            actor_loss_total = 0.0


            dtype = next(self.critic_engine.module.parameters()).dtype

            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}

                batch_tokens = batch_traj_process(
                    batch_slice['subtask'],
                    batch_slice['obs'],
                    batch_slice['action'],
                    self.low_engine.tokenizer
                ).to(self.low_engine.device)
                rewards, dones = self.prepare_tensor(batch_slice['reward'], batch_slice['done'])
                rewards = rewards.to(dtype)
                dones = dones.to(dtype)

                with torch.no_grad():
                    hidden_states, state_end_mask, action_end_mask = self.low_engine.get_hidden_states(batch_tokens)
                    target_vs, target_qsa = self.critic_engine.module.forward_hidden(hidden_states)

                    target_vs, _ = self.extract_valid(target_vs, action_end_mask)
                    target_qsa, _ = self.extract_valid(target_qsa, action_end_mask)

                vs, q_sa = self.critic_engine.module.forward_hidden(hidden_states)
                vs, _ = self.extract_valid(vs, action_end_mask)
                q_sa, _ = self.extract_valid(q_sa, action_end_mask)

                target = rewards[:, :-1] + (1 - dones[:, :-1]) * target_vs[:, 1:] * self.args['gama']
                target = target.to(dtype)
                q_loss = self.loss_fct(q_sa[:, :-1], target).to(dtype)

                diff = vs - target_qsa
                weight = torch.where(
                    diff < 0,
                    torch.ones_like(vs, dtype=dtype) * (1 - self.args['weight_tau']),
                    torch.ones_like(vs, dtype=dtype) * self.args['weight_tau']
                )
                v_loss = (weight * (diff ** 2)).mean().to(dtype)

                critic_loss = q_loss + v_loss
                self.critic_engine.backward(critic_loss)
                self.critic_engine.step()
                self.critic_engine.module.soft_update_target_critic(tau=self.args['tau'])

                action_log_probs, action_masks = self.low_engine.get_log_prob(batch_tokens)
                valid_action_log_probs = BC_AGENT.extract_valid_action_probs(
                    self, action_log_probs, action_masks, q_sa.size(1)
                ).to(dtype)

                if self.args['use_adv']:
                    with torch.no_grad():
                        adv = (q_sa - vs).to(dtype)
                    actor_loss = -(adv * valid_action_log_probs).mean()
                else:
                    actor_loss = -(q_sa.detach() * valid_action_log_probs).mean()

                self.low_engine.backward(actor_loss)
                self.low_engine.step()

                q_loss_total += q_loss.item() * len(batch_slice['obs'])
                v_loss_total += v_loss.item() * len(batch_slice['obs'])
                actor_loss_total += actor_loss.item() * len(batch_slice['obs'])

            return q_loss_total / total_size, v_loss_total / total_size, actor_loss_total / total_size


    def update(self):
        self.env = ScienceWorldEnv("", envStepLimit=self.args['env_step_limit'])
        self.task_names = self.env.getTaskNames()

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
            print(f"Epoch: {epoch}")
            for batch in dataloader:

                high_loss = self.update_policy(batch['high'], level="high")
             
          
                low_update_repeat = self.args.get("low_update_repeat", 1)
                for _ in range(low_update_repeat):

                    q_loss, actor_loss, v_loss = self.update_policy(batch['low'], level="low")
                
                self.critic.soft_update_target_critic(tau=self.args['tau'])

                if self.low_engine.local_rank == 0:
                    actor_val = actor_loss.item() if isinstance(actor_loss, torch.Tensor) else float(actor_loss)
                    q_val = q_loss.item() if isinstance(q_loss, torch.Tensor) else float(q_loss)
                    v_val = v_loss.item() if isinstance(v_loss, torch.Tensor) else float(v_loss)
                    step_val = self.low_global_step.item() if isinstance(self.low_global_step, torch.Tensor) else int(self.low_global_step)

                    self.low_writer.add_scalar('Loss/low/actor_loss', actor_val, step_val)
                    self.low_writer.add_scalar('Loss/low/critic_loss', q_val, step_val)
                    print(f"Low; step:{step_val}; actor_loss:{actor_val}; critic_loss:{q_val}; value_loss:{v_val}")

               
                if self.high_engine.local_rank == 0:
                    high_val = high_loss.item() if isinstance(high_loss, torch.Tensor) else float(high_loss)
                    step_val = self.high_global_step.item() if isinstance(self.high_global_step, torch.Tensor) else int(self.high_global_step)

                    self.high_writer.add_scalar('Loss/high/BC_loss', high_val, step_val)
                    print(f"High; step:{step_val}; BC_loss:{high_val}")
                    
               
                if step_val % self.args.get("log_freq", 1000) == 0:
                 
                if self.high_global_step.item() % self.args['eval_freq'] == 0 : 
                    BC_AGENT.save_off_policy(self, self.high_engine, self.high_global_step.item(), self.high_checkpoint_dir)

                    BC_AGENT.save_off_policy(self, self.low_engine, self.high_global_step.item(), self.low_checkpoint_dir)
                    avg_score = self.evaluate_online(self.env, num_episodes=2)
                    

                self.high_global_step += 1
                self.low_global_step += 1
            torch.cuda.empty_cache()

        BC_AGENT.save_off_policy(self, self.high_engine, self.high_global_step.item(), self.high_checkpoint_dir)
        BC_AGENT.save_off_policy(self, self.low_engine, self.high_global_step.item(), self.low_checkpoint_dir)
        BC_AGENT.save_critic(self, self.high_global_step.item(), self.critic_checkpoint_dir)

    def get_policy_q(self, batch_prompt, batch_obs_list, batch_action_list, level="low"):
        """
        Args:
            batch_prompt: List[str], 각 trajectory의 task_description
            batch_obs_list: List[List[str]], shape: (batch, steps+1)
            batch_action_list: List[List[int]], shape: (batch, steps)
            level: "high" or "low"
        Returns:
            q_values: Q(s, a~π), (batch, max_steps)
        """
        engine = self.low_engine if level == "low" else self.high_engine
        q_values = []

        for prompt, obs_list, action_list in zip(batch_prompt, batch_obs_list, batch_action_list):
            obs_list = obs_list[:-1]  
            traj_len = len(obs_list)
            q_list = []

   
            traj_token = engine.tokenizer(prompt, return_tensors='pt').to(engine.device)

            for t in range(traj_len):
                
                obs_token = engine.tokenizer(obs_list[t], return_tensors='pt').to(engine.device)
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], obs_token["input_ids"]], dim=1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], obs_token["attention_mask"]], dim=1)

               
                pi_action = engine.generate_action(copy.deepcopy(traj_token))[0]
                pi_action_token = engine.tokenizer(pi_action + engine.tokenizer.eos_token, return_tensors='pt').to(engine.device)
                input_token = {
                    "input_ids": torch.cat([traj_token["input_ids"], pi_action_token["input_ids"]], dim=1),
                    "attention_mask": torch.cat([traj_token["attention_mask"], pi_action_token["attention_mask"]], dim=1)
                }

                with torch.no_grad():
                    hidden_states = engine.base(**input_token, output_hidden_states=True).hidden_states[-1][:, -1]  
                    q_value, _ = self.critic_engine.target_critic(hidden_states)
                    q_list.append(q_value)

               
                action_token = engine.tokenizer(action_list[t] + engine.tokenizer.eos_token, return_tensors='pt').to(engine.device)
                traj_token["input_ids"] = torch.cat([traj_token["input_ids"], action_token["input_ids"]], dim=1)
                traj_token["attention_mask"] = torch.cat([traj_token["attention_mask"], action_token["attention_mask"]], dim=1)

            q_values.append(torch.cat(q_list))

 
        q_values = pad_sequence(q_values, batch_first=True, padding_value=0.0)
        return q_values
                
    def extract_valid(self, tensor, valid_mark):
        max_valid_len = valid_mark.sum(dim=1).max().item()
        if max_valid_len == 0:
            return torch.zeros((tensor.size(0), 0), device=tensor.device), valid_mark
        return tensor[:, :max_valid_len], valid_mark[:, :max_valid_len]

    

    def prepare_tensor(self, rewards, dones):
        """
        Args:
            rewards: List[(batch, steps)]
            dones: List[(batch, steps)]
        Returns:
            reward: tensor -> (batch, padding_steps)
            dones: tensor -> (batch, padding_steps)
        """
        reward_list = [torch.tensor(seq, dtype=torch.float, device=self.low_engine.device) for seq in rewards]
        done_list = [torch.tensor(seq, dtype=torch.float, device=self.low_engine.device) for seq in dones]
        
        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor = pad_sequence(done_list, batch_first=True, padding_value=0)
        
        return reward_tensor, done_tensor
    

    def evaluate_online(self, env, num_episodes=2):
    
        high_engine = self.high_engine
        low_engine = self.low_engine
        total_rewards = []

      
        task_names = [
            "boil", "change-the-state-of-matter-of", "chemistry-mix",
            "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
            "find-animal", "find-living-thing", "find-non-living-thing",
            "find-plant", "freeze", "grow-fruit", "grow-plant",
            "identify-life-stages-1", "identify-life-stages-2",
            "inclined-plane-determine-angle", "inclined-plane-friction-named-surfaces",
            "inclined-plane-friction-unnamed-surfaces", "lifespan-longest-lived",
            "lifespan-longest-lived-then-shortest-lived", "lifespan-shortest-lived",
            "measure-melting-point-known-substance", "measure-melting-point-unknown-substance",
            "melt", "mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant",
            "power-component", "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity", "test-conductivity-of-unknown-substances", "use-thermometer"
        ]

        vari_nums_list = [
            7, 7, 8, 9, 9, 10, 10, 10, 10, 7,
            10, 10, 3, 2, 0, 0, 0, 10, 10, 10,
            10, 0, 7, 0, 0, 3, 3, 10, 10, 10
        ]

        task_score = {}
        total_scores = []
        
        for ep in range(num_episodes):
            task_id = random.randrange(len(task_names))
            task_name = task_names[task_id]

            
            env.load(task_name)
            vari_ids = env.getVariationsDev()
            if not vari_ids:
                continue 

            vari_id = random.choice(vari_ids)

            score = self.eval_policy(task_id, vari_id, env)
            total_scores.append(score)

        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
        else:
            print("No valid tasks/variations evaluated.")
        return avg_score

    def eval_policy(self, task_id, vari_id, env):
        episode_steps = 0
        task_name = self.task_names[task_id]
        env.load(task_name, vari_id)
        obs, _= env.reset()
        task_description = env.taskdescription()
        print(f"task:{task_name}, vari:{vari_id}, {task_description}")
  
        high_traj_token = self.high_engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')

        traj_subtask, traj_group_action = [], []
        group_action = []
        done = False
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.high_engine.tokenizer(state, return_tensors='pt')
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.high_engine.generate_action(copy.deepcopy(high_traj_token))[0]
            subtask_token = self.high_engine.tokenizer(subtask + self.high_engine.tokenizer.eos_token, return_tensors='pt')
           
            traj_subtask.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.low_engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            raw_action_list = []
            group_reward, group_score = 0.0, 0.0
            
            while not subtask_done:
                episode_steps += 1
                obs_token = self.low_engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.low_engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)

                action, subtask_done = extract_action_done(raw_action)
                group_action.append(action)
                action_token = self.low_engine.tokenizer(raw_action+self.low_engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.env.step(action)
                group_reward += reward
                group_score += info['score']
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            traj_group_action.append(group_action)
        score = max(0, info['score'])
        return score
