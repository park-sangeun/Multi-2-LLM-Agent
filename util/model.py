import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"], use_auth_token=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.base = AutoModelForCausalLM.from_pretrained(args["model_name"], use_auth_token=True)
        if args["use_lora"]:
            lora_config = LoraConfig(
                r=16,
                target_modules=['v_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )

    def generate_action(self, state_ids):
        state_ids = state_ids.to(self.base.device)  # (batch, sqe_len)
        context_len = state_ids['input_ids'].size(1)
        outputs = self.base.generate(**state_ids, 
                                     max_new_tokens=self.args["max_new_tokens"],
                                     do_sample=self.args["do_sample"], 
                                     temperature=self.args["temperature"],
                                     pad_token_id=self.tokenizer.eos_token_id
                                     )
        
        raw_action = self.tokenizer.batch_decode(outputs[:, context_len:],
                                                 skip_special_tokens=True)


        return raw_action

    def get_log_prob(self, traj_token):
        output = self.base(
            input_ids=traj_token['input_ids'],
            attention_mask=traj_token['attention_mask']
        )
        logits = output.logits[:, :-1, :]   
        labels = traj_token['labels'][:, 1:]

        action_masks = (labels != -100).float()

        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0

        vocab_size = logits.size(-1)
        safe_labels = torch.clamp(safe_labels, min=0, max=vocab_size - 1)

        log_probs = torch.log_softmax(logits, dim=-1)

        action_log_probs = torch.gather(
            log_probs, 2, safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        return action_log_probs, action_masks
    

    def get_hidden_states(self, traj_token):
        """
        traj_token struct: prompt -> (state -> action)* -> padding
        Returns:
            hidden_states:(batch, seq_len, hidden_dim)
            state_end_mask: (batch, seq_len) (state end token set to 1, other is 0)
            action_end_mask: (batch, seq_len) (action end token set to 1, other is 0)
        """
        outputs = self.base(input_ids=traj_token['input_ids'], 
                           attention_mask=traj_token['attention_mask'],
                           output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        return hidden_states, traj_token["state_end_mask"], traj_token["action_end_mask"]

class HighPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        if args["use_lora_high"]:
            lora_config = LoraConfig(
                r=16,
                target_modules=[ 'v_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.base = get_peft_model(self.base, lora_config)

class LowPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        if args["use_lora_low"]:
            lora_config = LoraConfig(
                r=16,
                target_modules=['v_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.base = get_peft_model(self.base, lora_config)

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"], use_auth_token=True)
        self.base = AutoModelForCausalLM.from_pretrained(args["model_name"], use_auth_token=True)

        if args.get("use_lora_critic", False):
            lora_config = LoraConfig(
                r=16,
                target_modules=['v_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.base = get_peft_model(self.base, lora_config)

        hidden_dim = self.base.config.hidden_size

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.target_critic = copy.deepcopy(nn.ModuleDict({
            "base": self.base,
            "value_head": self.value_head,
            "q_head": self.q_head
        }))


    def forward(self, traj_token):
        """
        Args:
            traj_token: {
                "input_ids": [batch, seq_len],
                "attention_mask": [batch, seq_len],
            }
        Returns:
            values: (batch, seq_len)
            q_values: (batch, seq_len)
        """
        outputs = self.base(
            input_ids=traj_token['input_ids'],
            attention_mask=traj_token['attention_mask'],
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1] 
        values = self.value_head(hidden_states).squeeze(-1) 
        q_values = self.q_head(hidden_states).squeeze(-1)  

        return values, q_values
    def soft_update_target_critic(self, tau: float):
        """Target Critic ← Critic soft update"""
        assert 0.0 <= tau <= 1.0
        for target_param, param in zip(self.target_critic.parameters(), self.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def critic_forward(self, traj_token):
        _, q_values = self.forward(traj_token)
        return q_values


    @torch.no_grad()
    def target_critic_forward(self, traj_token):
        outputs = self.target_critic["base"](
            input_ids=traj_token['input_ids'],
            attention_mask=traj_token['attention_mask'],
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        q_values = self.target_critic["q_head"](hidden_states).squeeze(-1)
        return q_values


    def forward_hidden(self, hidden_states):
        values = self.value_head(hidden_states).squeeze(-1)
        q_values = self.q_head(hidden_states).squeeze(-1)
        return values, q_values
