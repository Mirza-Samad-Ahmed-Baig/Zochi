import logging
import sys
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

from pyreft import ReftTrainer, ReftModel, ReftTrainerForCausalLM

from csrf_router import CSRFRouter
from transformers import AdamW, get_scheduler

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class CSRFTrainerForCausalLM(ReftTrainerForCausalLM):
    """
    CSRF Trainer for Causal Language Modeling tasks.

    This trainer adapts the CSRFTrainer for causal language modeling.
    """
    
    def __init__(self, router: CSRFRouter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = router
        
        if hasattr(self.model, 'device_map'):
            device = self.model.device_map
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.router.to(device)
        logger.info(f"Router moved to device: {device}")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            # separate param groups: one for the base model, one for the router
            base_params = list(self.model.parameters())
            router_params = list(self.router.parameters())

            self.optimizer = AdamW(
                [
                    {"params": base_params, "lr": self.args.learning_rate}, 
                    {"params": router_params, "lr": self.args.learning_rate}, 
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay,
            )

        # Then the normal LR scheduler step
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )


    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        embeddings = model.model.get_input_embeddings()(input_ids)
        selected_embeddings = embeddings[:, 0, :]
        #selected_embeddings = selected_embeddings.float()
        gating_probs = self.router(selected_embeddings)
        active_subspaces = self.router.get_active_subspaces(gating_probs)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        with model.interventions_active(active_subspaces):
            outputs = model(batch_dict)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                logits = outputs.get("logits", None)
            else:
                loss = getattr(outputs, "loss", None)
                logits = getattr(outputs, "logits", None)

            if loss is None and logits is not None:
                # standard shifting for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

        if loss is None:
            raise ValueError("Loss is None... Check that 'labels' were passed and that the model supports returning a loss.")

        penalty_strength = 1e-3
        penalty = penalty_strength * torch.mean((gating_probs - 0.5)**2)

        total_loss = loss + penalty
        return total_loss
