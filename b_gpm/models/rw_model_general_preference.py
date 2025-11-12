# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import Optional
import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
from b_gpm.utils.logging import init_logger
from b_gpm.models.bayesian_types import BayesianEmbedding
import math

logger = init_logger(__name__)


# Construct reward model with a value head for sequence classification. (model also with a lm head)
def get_reward_model(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    init_prompt_head: bool = False,
    add_prompt_head: bool = False,
    is_general_preference: bool = False,
    is_bayesian_gpm: bool = False,
    value_head_dim: int = 2,
    **kwargs,
) -> nn.Module:
    """Get reward model with a value head(linear layer) and a lm head.

    Args:
        model_name_or_path (str): Path to pretrained model.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.
        init_value_head (bool, optional): Whether to initialize the value head weights. Defaults to False.
        is_general_preference (bool, optional): Whether to use General Preference model. Defaults to False (Bradley Terry model by default).
        is_bayesian_gpm (bool, optional): Whether to enable Bayesian GPM head. Implies is_general_preference.
        value_head_dim (int, optional): Dimension of value head for General Prefernce model. Ignored by the Bradley Terry model. Defaults to 2.

    Returns:
        nn.Module: pretrained transformer model.
    """

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if not load_in_4bit and getattr(config, "quantization_config", None):
        logger.info(
            "Detected quantization_config in checkpoint config, but load_in_4bit=False. "
            "Removing quantization metadata to keep loading unquantized base weights."
        )
        try:
            delattr(config, "quantization_config")
        except AttributeError:
            config.quantization_config = None
        if hasattr(config, "_pre_quantization_dtype"):
            delattr(config, "_pre_quantization_dtype")
    config._attn_implementation = (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )
    base_class = AutoModel._model_mapping[type(config)]
    base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
    cls_class = _get_reward_model(
        base_causal_class,
        base_class,
        is_general_preference or is_bayesian_gpm,
        is_bayesian_gpm,
        add_prompt_head,
        value_head_dim,
    )
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        **kwargs,
    )
    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module.to(torch.bfloat16)
                if "norm" in name:
                    module.to(torch.float32)
                if "value_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module.to(torch.bfloat16)

    if init_value_head:
        if dschf is not None:
            logger.info("Initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters(
                [model.value_head.weight], modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(
                        mean=0.0, std=1 / (config.hidden_size + 1)
                    )
        else:
            model.value_head.weight.data.normal_(
                mean=0.0, std=1 / (config.hidden_size + 1)
            )

    if init_prompt_head and add_prompt_head:
        if dschf is not None:
            logger.info("Initialize prompt_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters(
                [model.prompt_head.weight], modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    model.prompt_head.weight.data.normal_(
                        mean=0.0, std=1 / (config.hidden_size + 1)
                    )
        else:
            model.prompt_head.weight.data.normal_(
                mean=0.0, std=1 / (config.hidden_size + 1)
            )

    return model


def _get_reward_model(
    base_causal_model,
    base_llm_model,
    is_general_preference: bool = False,
    is_bayesian_gpm: bool = False,
    add_prompt_head: bool = False,
    value_head_dim: int = 2,
):
    class CustomRewardModel(base_causal_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.value_head_dim = value_head_dim
            use_general_head = is_general_preference or is_bayesian_gpm
            if not use_general_head:
                self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
            else:
                head_out_dim = value_head_dim * 2 if is_bayesian_gpm else value_head_dim
                self.value_head = nn.Linear(
                    config.hidden_size, head_out_dim, bias=False
                )
                if add_prompt_head:
                    self.prompt_head = nn.Linear(
                        config.hidden_size, value_head_dim // 2, bias=False
                    )

            self.is_general_preference = use_general_head
            self.is_bayesian_gpm = is_bayesian_gpm

            self.post_init()

        @staticmethod
        def _select_by_attention(
            tensor: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            if tensor.dim() != 3:
                raise ValueError("Expected 3D tensor for selection.")
            if tensor.size(1) != attention_mask.size(1):
                raise ValueError("Sequence length mismatch for attention mask.")
            seq_len = attention_mask.size(1)
            eos_indices = (
                seq_len - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            )
            gather_index = eos_indices.unsqueeze(-1).expand(-1, 1, tensor.size(-1))
            return tensor.gather(dim=1, index=gather_index).squeeze(1)

        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            if not self.is_general_preference:
                values = self.value_head(last_hidden_states).squeeze(-1)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = (
                        attention_mask.size(1)
                        - 1
                        - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    )
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
            else:
                reward = self._extract_preference_embeddings(
                    attention_mask, last_hidden_states
                )
                if return_output:
                    return reward, outputs
                return reward, None

        def create_skew_symmetric_block_matrix(
            self, dim, device, dtype, prompt_hidden_states
        ):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states. Only the relevant block diagonal parts are generated.

            Args:
            - dim: Dimension of the square matrix (must be even).
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim].

            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries.
            """
            if hasattr(self, "prompt_head"):
                batch_size = prompt_hidden_states.shape[0]

                # Ensure that dim is even, as we're creating blocks of size 2x2
                assert (
                    dim % 2 == 0
                ), "dim must be even for skew-symmetric block generation"

                # Pass through the linear layer to get the block diagonal entries (half of the matrix's off-diagonal blocks)
                block_values = self.prompt_head(prompt_hidden_states).view(
                    batch_size, dim // 2
                )
                hidden_dim = prompt_hidden_states.size(-1)
                block_values = torch.softmax(
                    block_values / math.sqrt(hidden_dim), dim=-1
                )
                block_values = block_values * block_values.shape[-1]

                # Create a batch of zero matrices [batch_size, dim, dim]
                batch_R_matrices = torch.zeros(
                    (batch_size, dim, dim), device=device, dtype=dtype
                )

                # Fill only the block diagonal entries with the learned values
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                    batch_R_matrices[:, i + 1, i] = block_values[
                        :, i // 2
                    ]  # Skew-symmetric condition
            else:
                raise AttributeError(
                    "prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization."
                )

            return batch_R_matrices

        def _extract_preference_embeddings(
            self, attention_mask: torch.Tensor, last_hidden_states: torch.Tensor
        ) -> torch.Tensor:
            values = self.value_head(last_hidden_states)
            if not self.is_bayesian_gpm:
                if self.training:
                    return values[:, -1, :]
                return self._select_by_attention(values, attention_mask)

            mu, logvar = torch.chunk(values, 2, dim=-1)
            if self.training:
                mu = mu[:, -1, :]
                logvar = logvar[:, -1, :]
            else:
                mu = self._select_by_attention(mu, attention_mask)
                logvar = self._select_by_attention(logvar, attention_mask)

            # L2 normalize the mean embedding to maintain unit length constraint
            mu = nn.functional.normalize(mu, p=2, dim=-1)

            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            # Sample and normalize to maintain unit length constraint
            sample = mu + eps * std
            sample = nn.functional.normalize(sample, p=2, dim=-1)

            return BayesianEmbedding(sample=sample, mean=mu, logvar=logvar)

    return CustomRewardModel
