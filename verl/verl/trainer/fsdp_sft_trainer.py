# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os
from pathlib import Path

# Load environment variables from .env file (for WANDB_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    # Look for .env in self_play root (3 levels up from verl/verl/trainer/)
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import (
    entropy_from_logits,
    entropy_from_logits_with_chunking,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils.wikitext_eval import compute_wikitext_loss
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        # Use separate val_batch_size_per_gpu if specified, otherwise fall back to micro_batch_size_per_gpu
        val_batch_size_per_gpu = getattr(config.data, 'val_batch_size_per_gpu', config.data.micro_batch_size_per_gpu)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                # Option to disable fused linear+CE while keeping other Liger optimizations
                # (RMSNorm, SwiGLU, RoPE). Set liger_fused_linear_ce=False to use standard CE.
                use_fused_linear_ce = self.config.model.get("liger_fused_linear_ce", True)
                _apply_liger_kernel_to_instance(
                    model=self.model,
                    fused_linear_cross_entropy=use_fused_linear_ce,
                )

            # Optionally freeze lm_head (useful with Liger + sample weighting)
            # When frozen, we can use the simpler hidden_states hook for weighting
            # instead of the more expensive summon_full_params approach
            if self.config.model.get("freeze_lm_head", False):
                if hasattr(self.model, 'lm_head'):
                    for param in self.model.lm_head.parameters():
                        param.requires_grad = False
                    print("[INFO] lm_head frozen (freeze_lm_head=True)")

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        # Disable KV cache to avoid issues with gradient checkpointing and dynamic shapes
        # See: https://github.com/huggingface/transformers/pull/36610
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _forward_micro_batch(self, batch, temperature=1.0, calculate_entropy=False):
        """
        Forward pass to get log probs and entropy (for PPO loss).
        Adapted from verl's dp_actor._forward_micro_batch() but simplified for pre-tokenized data.

        Args:
            batch: Dict with input_ids, attention_mask, position_ids
            temperature: Temperature for scaling logits
            calculate_entropy: Whether to compute entropy

        Returns:
            (entropy, log_probs) where:
                entropy: (bs, seq_len-1) if calculate_entropy else None
                log_probs: (bs, seq_len-1)
        """
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # Forward pass
            output = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = output.logits  # (bs, seq_len, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # (bs, seq_len-1, vocab_size)
            shift_labels = input_ids[:, 1:].contiguous()   # (bs, seq_len-1)

            # Compute log probs using verl's function
            from verl.utils.torch_functional import logprobs_from_logits
            log_probs = logprobs_from_logits(shift_logits, shift_labels, inplace_backward=False)  # (bs, seq_len-1)

            # Compute entropy if requested
            entropy = None
            if calculate_entropy:
                if not self.config.trainer.get("entropy_checkpointing", False):
                    entropy = verl_F.entropy_from_logits(shift_logits)  # (bs, seq_len-1)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, shift_logits)

        return entropy, log_probs

    def _compute_ppo_loss_and_backward(self, batch, do_backward=True):
        """
        Compute PPO policy gradient loss (analogous to dp_actor.update_policy).
        Uses verl's compute_grpo_outcome_advantage and compute_policy_loss.

        Args:
            batch: Dict with old_log_probs, loss_mask, sample_weight, group_index
            do_backward: Whether to call backward on the loss

        Returns:
            loss: Scalar loss tensor
        """
        from verl.trainer.ppo.core_algos import compute_policy_loss, compute_grpo_outcome_advantage, agg_loss

        # Get config parameters
        clip_ratio = self.config.trainer.get('ppo_clip_ratio', 0.2)
        clip_ratio_low = self.config.trainer.get('ppo_clip_ratio_low', clip_ratio)
        clip_ratio_high = self.config.trainer.get('ppo_clip_ratio_high', clip_ratio)
        entropy_coeff = self.config.trainer.get('entropy_coeff', 0.0)
        temperature = 1.0  # Can be made configurable if needed

        # Extract data from batch
        old_log_probs = batch.pop("old_log_probs").to(self.device_name)  # (bs, seq_len)
        loss_mask = batch.pop("loss_mask")[:, 1:].to(self.device_name)  # (bs, seq_len-1) - response mask
        sample_weight = batch.pop("sample_weight").to(self.device_name)  # (bs,) - rewards
        group_indices = batch.pop("group_index").cpu().numpy()  # (bs,) - for GRPO

        # Trim old_log_probs to match shifted sequence length
        old_log_probs = old_log_probs[:, :-1]  # (bs, seq_len-1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # 1. Forward pass to get new log probs
            calculate_entropy = (entropy_coeff > 0)
            entropy, new_log_probs = self._forward_micro_batch(
                batch, temperature=temperature, calculate_entropy=calculate_entropy
            )

            # 2. Compute token-level rewards
            # IMPORTANT: GRPO advantage estimators expect terminal rewards (reward only on last token),
            # NOT broadcast rewards. The compute_grpo_outcome_advantage function sums token_level_rewards
            # across the sequence (token_level_rewards.sum(dim=-1)), so broadcasting the scalar reward
            # to all tokens would inflate the sum by ~300x (sequence length).
            #
            # This differs from the SFT loss convention where sample_weight is broadcast to all tokens
            # (see _compute_loss_and_backward), because SFT applies per-token cross-entropy loss,
            # while GRPO operates on sequence-level outcome rewards.
            B = sample_weight.size(0)
            seq_len = loss_mask.size(1)
            token_rewards = torch.zeros(B, seq_len, device=sample_weight.device, dtype=sample_weight.dtype)

            # Place scalar reward on the last response token (terminal reward)
            for i in range(B):
                response_length = loss_mask[i].sum()
                if response_length > 0:
                    last_token_idx = int(response_length) - 1
                    token_rewards[i, last_token_idx] = sample_weight[i]  # (bs, seq_len-1)

            # 3. Compute GRPO advantages using verl's function
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=token_rewards,
                response_mask=loss_mask,
                index=group_indices,
                norm_adv_by_std_in_grpo=True,  # Use std normalization (GRPO-style)
            )

            # 4. Compute PPO policy loss using verl's function
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=old_log_probs,
                log_prob=new_log_probs,
                advantages=advantages,
                response_mask=loss_mask,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                loss_agg_mode="token-mean",
            )

            # 5. Add entropy regularization
            if entropy_coeff > 0:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=loss_mask, loss_agg_mode="token-mean")
                policy_loss = pg_loss - entropy_coeff * entropy_loss
            else:
                policy_loss = pg_loss

            # 6. Backward
            if do_backward:
                policy_loss.backward()

        return policy_loss

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        # Optional per-sample scalar weight (GRPO-style weighting). If present, broadcast to tokens.
        sample_weight = batch.pop("sample_weight", None)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Get entropy regularization coefficient
        entropy_coeff = self.config.trainer.get('entropy_coeff', 0.0)

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # Check if we should use Liger's fused linear+cross-entropy kernel
            # Requires: use_liger=True AND liger_fused_linear_ce=True AND no sequence parallel AND entropy=0
            use_liger_config = self.config.model.get("use_liger", False)
            liger_fused_linear_ce = self.config.model.get("liger_fused_linear_ce", True)
            use_liger = use_liger_config and liger_fused_linear_ce and not use_sp and entropy_coeff == 0

            # Debug logging for Liger activation status (only on rank 0)
            if self.config.trainer.get('local_rank', 0) == 0 and hasattr(self, '_liger_debug_logged') is False:
                self._liger_debug_logged = True
                if not use_liger:
                    reasons = []
                    if not use_liger_config:
                        reasons.append("model.use_liger=False")
                    if not liger_fused_linear_ce:
                        reasons.append("model.liger_fused_linear_ce=False")
                    if use_sp:
                        reasons.append("sequence_parallel=True")
                    if entropy_coeff != 0:
                        reasons.append(f"entropy_coeff={entropy_coeff} (must be 0)")
                    print(f"\n{'='*80}\n[LIGER DISABLED] Reasons: {', '.join(reasons)}\nUsing standard PyTorch cross-entropy (materializes logits)\n{'='*80}\n")

            if use_liger:
                # ========== LIGER FUSED LINEAR+CROSS-ENTROPY PATH ==========
                # Avoids materializing (B, L, V) logits tensor, saving ~13GB memory

                B = input_ids.size(0)
                L = input_ids.size(1)
                loss_mask_2d = loss_mask.view(B, L - 1)  # (B, L-1)

                # Prepare labels: mark non-assistant tokens with -100 (ignored by Liger)
                labels = input_ids.clone()
                labels[:, :-1][loss_mask_2d == 0] = -100  # Mask non-assistant tokens
                labels[:, -1] = -100  # Last token has no target

                # Check if we need sample weighting
                freeze_lm_head = self.config.model.get("freeze_lm_head", False)

                if sample_weight is not None:
                    # ========== SAMPLE WEIGHTING PATH ==========
                    # Two options depending on whether lm_head is frozen

                    if freeze_lm_head:
                        # Option B: lm_head frozen, use hidden_states hook
                        # Decoder gradients are weighted, lm_head has no gradient
                        def make_weight_hook(weights):
                            def hook(grad):
                                # grad: (B, L, H)
                                B = weights.size(0)
                                w = weights.view(B, 1, 1).to(grad.device).to(grad.dtype)
                                return grad * w
                            return hook

                        # Get hidden states from decoder (batched, efficient)
                        decoder_output = self.fsdp_model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False
                        )
                        hidden_states = decoder_output[0]  # (B, L, H)

                        # Register hook to weight gradients during backward
                        hook_handle = hidden_states.register_hook(make_weight_hook(sample_weight))

                        # Call full model with labels to trigger Liger's fused path
                        # Note: This does redundant decoder forward, but keeps code simple
                        # The hook on hidden_states will apply weights during backward
                        output = self.fsdp_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                            skip_logits=True,
                            use_cache=False
                        )
                        base_loss = output.loss

                        # Compute weighted average for correct loss value (for logging)
                        # Note: gradients are handled by hook, this is just for the returned loss
                        if self.config.data.balance_dp_token:
                            dp_size = torch.distributed.get_world_size()
                        else:
                            dp_size = 1
                        loss = base_loss * dp_size

                        if do_backward:
                            loss.backward()
                        hook_handle.remove()
                        return loss

                    else:
                        # Option A (default): Use summon_full_params for per-sample fused CE
                        # Batched decoder forward, sequential fused kernel calls
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

                        # 1. Batched forward through decoder (efficient)
                        decoder_output = self.fsdp_model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False
                        )
                        hidden_states = decoder_output[0]  # (B, L, H)
                        H = hidden_states.size(-1)

                        # Prepare shifted hidden states and labels for next-token prediction
                        shift_hidden = hidden_states[:, :-1, :].contiguous()  # (B, L-1, H)
                        shift_labels = labels[:, 1:].contiguous()  # (B, L-1)

                        # 2. Sequential per-sample fused CE with sample weights
                        # Use summon_full_params to access unsharded lm_head.weight
                        fused_ce = LigerFusedLinearCrossEntropyLoss(
                            ignore_index=-100,
                            reduction='mean',  # Mean over valid tokens per sample
                        )

                        total_weighted_loss = torch.tensor(0.0, device=input_ids.device)
                        total_weight = torch.tensor(0.0, device=input_ids.device)

                        with FSDP.summon_full_params(self.fsdp_model, writeback=True, recurse=True):
                            lm_head_weight = self.fsdp_model.lm_head.weight  # (V, H)

                            for i in range(B):
                                sample_hidden = shift_hidden[i].reshape(-1, H)  # (L-1, H)
                                sample_labels = shift_labels[i].reshape(-1)  # (L-1,)

                                sample_loss = fused_ce(
                                    lm_head_weight,
                                    sample_hidden,
                                    sample_labels,
                                )

                                w = sample_weight[i]
                                total_weighted_loss = total_weighted_loss + w * sample_loss
                                total_weight = total_weight + w.abs()  # Use abs for normalization

                        # Normalize by total weight
                        base_loss = total_weighted_loss / (total_weight + 1e-8)

                        if self.config.data.balance_dp_token:
                            dp_size = torch.distributed.get_world_size()
                        else:
                            dp_size = 1
                        loss = base_loss * dp_size

                        if do_backward:
                            loss.backward()
                        return loss

                else:
                    # ========== NO SAMPLE WEIGHTING (simple path) ==========
                    # Use Liger's monkey-patched forward directly
                    output = self.fsdp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                        skip_logits=True,
                        use_cache=False
                    )
                    base_loss = output.loss

                    if self.config.data.balance_dp_token:
                        dp_size = torch.distributed.get_world_size()
                    else:
                        dp_size = 1

                    loss = base_loss * dp_size
                    if do_backward:
                        loss.backward()
                    return loss

            elif not use_sp:
                # ========== STANDARD PATH (original code) ==========
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()

                # Compute entropy regularization (before flattening logits)
                entropy_bonus = torch.tensor(0.0, device=input_ids.device)
                if entropy_coeff > 0:
                    B = shift_logits.size(0)
                    seq_len = shift_logits.size(1)

                    # Memory-efficient entropy: use checkpointing to recompute during backward
                    # This avoids storing softmax probabilities (6+ GB) in memory
                    def compute_entropy(logits_2d, mask):
                        """Compute entropy in chunks without float cast (stays in bfloat16)"""
                        chunk_size = 512
                        entropy = torch.zeros(logits_2d.shape[0], device=logits_2d.device, dtype=logits_2d.dtype)
                        for i in range(0, logits_2d.shape[0], chunk_size):
                            logits_chunk = logits_2d[i : i + chunk_size]
                            # Keep in bfloat16 - no .float() cast
                            pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
                            entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
                            entropy[i : i + chunk_size] = entropy_chunk
                        masked = entropy * mask
                        return torch.sum(masked) / (torch.sum(mask) + 1e-8)

                    # Reshape and checkpoint the entropy computation
                    shift_logits_2d = shift_logits.view(-1, self.model.config.vocab_size)
                    mask = loss_mask.to(shift_logits_2d.device)
                    # Use checkpoint to recompute during backward, saving memory
                    entropy_bonus = torch.utils.checkpoint.checkpoint(
                        compute_entropy, shift_logits_2d, mask, use_reentrant=False
                    )

                # Flatten the tokens for cross-entropy
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
                if sample_weight is not None:
                    # sample_weight: (B,) => expand to token-level weights
                    B = input_ids.size(0)
                    token_weights = sample_weight.to(loss.device).view(B, 1).expand(B, input_ids.size(1) - 1)
                    loss = loss.view(B, -1) * token_weights
                    loss = loss.reshape(-1)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                # Note: Entropy regularization for SP path would require gathering logits across ranks,
                # which adds complexity. For now, we only apply entropy bonus in non-SP path.
                entropy_bonus = torch.tensor(0.0, device=input_ids.device)

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask
                if sample_weight is not None:
                    # For SP path, full_loss has shape (B, L-1) after reconstruction
                    B = input_ids.shape[0]
                    token_weights = sample_weight.to(full_loss.device).view(B, 1).expand_as(full_loss)
                    loss = (full_loss * token_weights).reshape(-1) * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            # Apply entropy regularization: subtract entropy bonus to encourage higher entropy
            if entropy_coeff > 0:
                # Scale entropy_bonus by dp_size to match cross-entropy loss scaling
                entropy_bonus_scaled = entropy_bonus * dp_size
                loss = loss - entropy_coeff * entropy_bonus_scaled

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0

        # Choose loss function based on config
        use_ppo_loss = self.config.trainer.get('use_ppo_loss', False)

        for micro_batch in micro_batches:
            if use_ppo_loss:
                loss = self._compute_ppo_loss_and_backward(batch=micro_batch) / n_micro_batches
            else:
                loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def _offload_optimizer_to_cpu(self):
        """Move optimizer state to CPU to free GPU memory during evaluation"""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

    def _load_optimizer_to_gpu(self):
        """Move optimizer state back to GPU after evaluation"""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device_name)

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        # Choose loss function based on config
        use_ppo_loss = self.config.trainer.get('use_ppo_loss', False)

        with torch.no_grad():
            if use_ppo_loss:
                loss = self._compute_ppo_loss_and_backward(batch, do_backward=False)
            else:
                loss = self._compute_loss_and_backward(batch, do_backward=False)

            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        # Perform validation before training if configured
        if self.config.trainer.get("val_before_train", False) and global_step == 0:
            log_with_rank(
                "Running validation before training...",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )
            # Optionally offload optimizer state to CPU for larger eval batches
            offload_optimizer_during_eval = getattr(self.config.trainer, 'offload_optimizer_during_eval', True)
            if offload_optimizer_during_eval:
                self._offload_optimizer_to_cpu()
                torch.distributed.barrier()

            # Perform validation
            val_losses = []
            for val_data in tqdm(
                self.val_dataloader,
                desc="Initial Validation",
                disable=rank != 0,
            ):
                val_batch_size_per_gpu = getattr(self.config.data, 'val_batch_size_per_gpu', self.config.data.micro_batch_size_per_gpu)
                val_data = TensorDict(val_data, batch_size=val_batch_size_per_gpu).to(
                    self.device_name
                )
                val_loss = self.validation_step(val_data)
                val_losses.append(val_loss)
            # Wikitext evaluation (if enabled) - runs on all ranks with data parallelism
            wikitext_metrics = None
            if self.config.trainer.get("eval_wikitext", False):
                wikitext_path = self.config.trainer.get("wikitext_path", "data/eval/wikitext_sample.txt")
                max_seq_length = self.config.trainer.get("wikitext_max_seq_length", 2048)
                wikitext_batch_size = self.config.trainer.get("wikitext_batch_size", 32)

                if rank == 0:
                    print(f"Running wikitext evaluation on {wikitext_path}...")
                try:
                    wikitext_metrics = compute_wikitext_loss(
                        model=self.fsdp_model,
                        tokenizer=self.tokenizer,
                        wikitext_path=wikitext_path,
                        max_seq_length=max_seq_length,
                        batch_size=wikitext_batch_size,
                        device=self.device_name,
                        distributed=True,
                        rank=rank,
                        world_size=world_size,
                    )
                    if rank == 0:
                        print(f"Wikitext metrics: {wikitext_metrics}")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Wikitext evaluation failed: {e}")

            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}

                if wikitext_metrics is not None:
                    metric.update(wikitext_metrics)

                tracking.log(data=metric, step=global_step)
                last_valid_metric = metric
                print(f"Initial validation metrics: {metric}")
            torch.distributed.barrier()

            # Reload optimizer state back to GPU for training
            if offload_optimizer_during_eval:
                self._load_optimizer_to_gpu()
                torch.distributed.barrier()

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Optionally offload optimizer state to CPU to free GPU memory for larger eval batches
                    offload_optimizer_during_eval = getattr(self.config.trainer, 'offload_optimizer_during_eval', True)
                    if offload_optimizer_during_eval:
                        self._offload_optimizer_to_cpu()
                        torch.distributed.barrier()  # Ensure all ranks finish offloading

                    # Perform validation
                    val_losses = []
                    for val_data in tqdm(
                        self.val_dataloader,
                        desc="Validation",
                        disable=rank != 0,
                    ):
                        val_batch_size_per_gpu = getattr(self.config.data, 'val_batch_size_per_gpu', self.config.data.micro_batch_size_per_gpu)
                        val_data = TensorDict(val_data, batch_size=val_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    # Wikitext evaluation (if enabled) - runs on all ranks with data parallelism
                    wikitext_metrics = None
                    if self.config.trainer.get("eval_wikitext", False):
                        wikitext_path = self.config.trainer.get("wikitext_path", "data/eval/wikitext_sample.txt")
                        max_seq_length = self.config.trainer.get("wikitext_max_seq_length", 2048)
                        wikitext_batch_size = self.config.trainer.get("wikitext_batch_size", 32)

                        if rank == 0:
                            print(f"Running wikitext evaluation on {wikitext_path}...")
                        try:
                            wikitext_metrics = compute_wikitext_loss(
                                model=self.fsdp_model,
                                tokenizer=self.tokenizer,
                                wikitext_path=wikitext_path,
                                max_seq_length=max_seq_length,
                                batch_size=wikitext_batch_size,
                                device=self.device_name,
                                distributed=True,
                                rank=rank,
                                world_size=world_size,
                            )
                            if rank == 0:
                                print(f"Wikitext metrics: {wikitext_metrics}")
                        except Exception as e:
                            if rank == 0:
                                print(f"Warning: Wikitext evaluation failed: {e}")

                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}

                        if wikitext_metrics is not None:
                            metric.update(wikitext_metrics)

                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                    # Reload optimizer state back to GPU for training
                    if offload_optimizer_during_eval:
                        self._load_optimizer_to_gpu()
                        torch.distributed.barrier()  # Ensure all ranks finish reloading

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    chat_template_path = config.model.get("chat_template_path", None)
    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code,
        chat_template_path=chat_template_path,
    )
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer, full_config=config)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer, full_config=config)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, full_config=None):
    """Create a dataset.

    Args:
        data_paths: Path(s) to dataset files
        data_config: Data configuration (config.data)
        tokenizer: Tokenizer instance
        full_config: Optional full config object (for datasets that need access to trainer config)
    """
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    # For custom datasets (like PreTokenizedSFTDataset), pass full config if available
    # so they can access trainer settings like use_ppo_loss
    if full_config is not None and data_config.custom_cls.get("path", None):
        # Merge data and trainer configs for custom datasets
        from omegaconf import OmegaConf
        merged_config = OmegaConf.create({
            **OmegaConf.to_container(data_config, resolve=True),
            "trainer": OmegaConf.to_container(full_config.trainer, resolve=True) if hasattr(full_config, "trainer") else {}
        })
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=merged_config)
    else:
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
