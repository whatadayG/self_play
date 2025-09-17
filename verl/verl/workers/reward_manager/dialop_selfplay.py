# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register


@register("dialop_selfplay")
class DialopSelfPlayRewardManager:
    """
    A reward manager for dialop self-play that assigns shared rewards to both agents.
    
    In dialop self-play:
    - Two agents (both the same LLM) engage in a cooperative conversation
    - Both agents receive the same reward based on the final outcome
    - Rewards are assigned to the last token of each agent's responses
    
    Args:
        tokenizer: The tokenizer for decoding responses
        num_examine: Number of conversations to print for debugging
        compute_score: Custom scoring function (optional, uses data from interaction)
        reward_fn_key: Key to access reward data (default: "reward_model")
    """
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="reward_model", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self._already_printed = 0
        
    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards for both agents in self-play conversations.
        
        For self-play, we need to:
        1. Identify which tokens belong to which agent
        2. Assign the shared reward to both agents' final tokens
        3. Handle multi-turn conversations properly
        """
        # Check if rewards are already computed
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
                
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Get the final reward from the interaction
            reward_info = data_item.non_tensor_batch.get(self.reward_fn_key, {})
            
            # Handle both dict and scalar rewards
            if isinstance(reward_info, dict):
                # Prefer normalized reward if available
                reward = reward_info.get("normalized_reward", reward_info.get("reward", 0.0))
                game_info = reward_info.get("game_info", {})
                
                for key, value in reward_info.items():
                    reward_extra_info[key].append(value)
            else:
                reward = float(reward_info) if reward_info is not None else 0.0
                game_info = {}
            
            # For self-play, assign reward to the last valid token
            # This represents the final outcome of the collaborative game
            valid_length = int(valid_response_lengths[i].item())
            if valid_length > 0:
                reward_tensor[i, valid_length - 1] = reward
                
            # Optionally print some examples for debugging
            if self._already_printed < self.num_examine:
                response_ids = data_item.batch["responses"]
                valid_response_ids = response_ids[:valid_length]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                
                print(f"\n[Example {self._already_printed + 1}]")
                print("[Prompt]", prompt_str[:200] + "..." if len(prompt_str) > 200 else prompt_str)
                print("[Response]", response_str[:200] + "..." if len(response_str) > 200 else response_str)
                print("[Reward]", reward)
                if game_info:
                    print("[Game Info]", game_info)
                print("-" * 80)
                
                self._already_printed += 1
                
        # Store accuracy metric for tracking
        rewards = []
        for i in range(len(data)):
            reward_info = data[i].non_tensor_batch.get(self.reward_fn_key, {})
            if isinstance(reward_info, dict):
                reward = reward_info.get("normalized_reward", reward_info.get("reward", 0.0))
            else:
                reward = float(reward_info) if reward_info is not None else 0.0
            rewards.append(reward)
            
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor