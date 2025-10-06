import deepspeed
import pandas as pd
import random
import torch
from util.model import Policy, HighPolicy, LowPolicy
from alg.bc import Agent
from scienceworld import ScienceWorldEnv
import copy
import os
import numpy as np
from prompt.inst import high_prompt, low_prompt
from util.extract import extract_action_done

class EvalAgent:
    def __init__(self, args):
        self.args = args
        self.high_policy = HighPolicy(args)
        self.low_policy = LowPolicy(args)
        self.high_engine, *_ = deepspeed.initialize(
            model=self.high_policy,
            model_parameters=[{"params": [p for p in self.high_policy.base.parameters() if p.requires_grad],
                               "lr": args["lr_high"]}],
            config=args["ds_config"]
        )

        self.low_engine, *_ = deepspeed.initialize(
            model=self.low_policy,
            model_parameters=[{"params": [p for p in self.low_policy.base.parameters() if p.requires_grad],
                               "lr": args["actor_lr"]}],
            config=args["ds_config"]
        )
	    self.high_checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/high"
        self.low_checkpoint_dir = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/low"

        self.eval_env = ScienceWorldEnv("", envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()
        print("Load")
        Agent.load_low_policy(self, self.low_checkpoint_dir)
	    Agent.load_high_policy(self, self.high_checkpoint_dir)


    def evaluate_online(self, num_episodes=10, dev_or_test="dev"):
    
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

       
        if dev_or_test == "test":
            vari_nums_list = [
                9, 9, 8, 9, 9, 10, 10, 10, 10, 9,
                10, 10, 5, 4, 0, 0, 0, 10, 10, 10,
                10, 0, 9, 0, 0, 5, 5, 10, 10, 10
            ]
        elif dev_or_test == "dev":
            vari_nums_list = [
                7, 7, 8, 9, 9, 10, 10, 10, 10, 7,
                10, 10, 3, 2, 0, 0, 0, 10, 10, 10,
                10, 0, 7, 0, 0, 3, 3, 10, 10, 10
            ]
        else: 
            vari_nums_list = [
                14, 14, 16, 18, 18, 120, 120, 120, 120, 14,
                62, 62, 6, 4, 0, 0, 0, 62, 62, 62,
                0, 0, 14, 0, 0, 8, 8, 120, 120, 120
            ]


        task_score = {}
        total_scores = []
        failure = 0
        total_task = 0
        for ep in range(num_episodes):
          
            task_id = random.choice([29, 20, 21])
            task_name = task_names[task_id]
           
            self.eval_env.load(task_name)
            if dev_or_test == "test":
                vari_ids = self.eval_env.getVariationsTest()
            elif dev_or_test == "dev":
                vari_ids = self.eval_env.getVariationsDev()
            else:
                vari_ids = self.eval_env.getVariationsTrain()

            if not vari_ids:
                continue 
            vari_id = random.choice(vari_ids)

            score = self.eval_policy(task_id, vari_id)
            if score == 0:
                failure +=1
                total_task +=1
            else:
                total_scores.append(score)
                total_task +=1

            print(f"[Episode {ep+1}] Task: {task_name}, Variation: {vari_id}, Score: {score}")

        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)

            mean = np.mean(total_scores)
            std = np.std(total_scores)
            print(f"\n=== Final Result over {num_episodes} episodes: {avg_score:.3f} ===")
            print(f"\nFailure: {failure} per Total: {total_task}")
            print(f"{total_scores} \n Mean: {mean} +- {std}")
        else:
            print("No valid tasks/variations evaluated.")
        return avg_score

    
    def eval_policy(self, task_id, vari_id):
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        obs, _= self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        print(f"task:{task_name}, vari:{vari_id}, {task_description}, obs: {obs}")
   
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
                obs_, reward, done, info = self.eval_env.step(action)
                reward = reward/100
                score = info['score']/100

                print(f"[Step {episode_steps}] Action: {action}, New Obs: {obs_}, Reward: {reward}, Score: {score}")

                
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            traj_group_action.append(group_action)
    

        print("subtask: ", traj_subtask)
        print("group action:", traj_group_action)
        score = max(0, score)
        print(f"score: {score}")
        return score

    def data_collect(self, task_id, vari_id, high_data_container, low_data_container):
        """
        Args:
            high_data_container:{
                'task_description': [task_num,],
                'obs': [task_num, groups+1],
                'subtask': [task_num, groups],
                'reward': [task_num, groups],
                'score': [task_num, groups],
                'done': [task_num, groups]
            }
            low_data_container:{
                'subtask':[subtask_nums, steps],
                'obs':[subtask_nums, steps+1],
                'action':[subtask_nums, steps],
                'reward':[subtask_nums, steps],
                'score':[subtask_nums, steps],
                'done':[subtask_nums, steps]
            }
            score_threshold:[min, max]
        """
        high_obs_traj, high_subtask_traj, high_reward_traj, high_score_traj, high_done_traj = [], [], [], [], []
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
        task_description = self.eval_env.taskdescription()
        print(task_id, vari_id, task_description)
        obs, _= self.eval_env.reset()
        high_traj_token = self.high_engine.tokenizer(high_prompt + " " + task_description, return_tensors='pt')

        done = False
        group_action = []
        while not done:
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_token = self.high_engine.tokenizer(state, return_tensors='pt')
            high_obs_traj.append(state)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], state_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], state_token["attention_mask"]], dim = 1)
            subtask = self.high_engine.generate_action(copy.deepcopy(high_traj_token))[0]
            subtask_token = self.high_engine.tokenizer(subtask + self.high_engine.tokenizer.eos_token, return_tensors='pt')
            print("subtask:", subtask)
            high_subtask_traj.append(subtask)
            high_traj_token["input_ids"] = torch.cat([high_traj_token["input_ids"], subtask_token["input_ids"]], dim = 1)
            high_traj_token["attention_mask"] = torch.cat([high_traj_token["attention_mask"], subtask_token["attention_mask"]], dim = 1)

            low_group_token = self.low_engine.tokenizer(low_prompt + " Subtask: " + subtask, return_tensors='pt')
            subtask_done = False
            group_action = []
            group_reward, group_score = 0.0, 0.0
            raw_action_list = []
            low_obs_traj, low_reward_traj, low_score_traj, low_done_traj = [], [], [], []
            low_init_obs = obs
            while not subtask_done:
                low_obs_traj.append("Obs: "+obs)
                episode_steps += 1
                obs_token = self.low_engine.tokenizer("Obs: "+obs, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], obs_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], obs_token["attention_mask"]], dim = 1)
                raw_action = self.low_engine.generate_action(copy.deepcopy(low_group_token))[0]
                raw_action_list.append(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                low_done_traj.append(subtask_done)
                group_action.append(action)
                action_token = self.low_engine.tokenizer(raw_action+self.low_engine.tokenizer.eos_token, return_tensors='pt')
                low_group_token["input_ids"] = torch.cat([low_group_token["input_ids"], action_token["input_ids"]], dim = 1)
                low_group_token["attention_mask"] = torch.cat([low_group_token["attention_mask"], action_token["attention_mask"]], dim = 1)
                obs_, reward, done, info = self.eval_env.step(action)
                group_reward += reward/100
                group_score += info['score']/100
                obs = obs_
                if episode_steps == self.args['env_step_limit']:
                    done = True
                    break
            print("group action: ", raw_action_list, info['score'])
  
            
            low_obs_traj.append("Obs: "+obs)
            low_data_container['subtask'].append(low_prompt + " Subtask: " + subtask)
            low_data_container['obs'].append(low_obs_traj)
            low_data_container['action'].append(raw_action_list)
            low_data_container['done'].append(low_done_traj)
            
            high_reward_traj.append(group_reward)
            high_score_traj.append(group_score)
            high_done_traj.append(False if episode_steps==self.args['env_step_limit'] else done)
        state = f"Group action: {group_action}. Current observation: {obs}"
        high_obs_traj.append(state)
        high_data_container['task_description'].append(high_prompt + " " + task_description)
        high_data_container['obs'].append(high_obs_traj)
        high_data_container['subtask'].append(high_subtask_traj)
        high_data_container['done'].append(high_done_traj)
        high_data_container['score'].append(high_score_traj)
        high_data_container['reward'].append(high_reward_traj)


    
