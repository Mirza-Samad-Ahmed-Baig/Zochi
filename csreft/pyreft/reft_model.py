import pyvene as pv
from contextlib import contextmanager


def count_parameters(model):
    """Count parameters of a model that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ReftModel(pv.IntervenableModel):
    """
    Base model for Reft methods.
    Wraps a Hugging Face transformers model to handle HF-specific calls.
    """
    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)

    @staticmethod
    def _convert_to_reft_model(intervenable_model):
        reft_model = ReftModel(intervenable_model.config, intervenable_model.model)
        for attr in vars(intervenable_model):
            setattr(reft_model, attr, getattr(intervenable_model, attr))
        return reft_model

    @staticmethod
    def load(*args, **kwargs):
        model = pv.IntervenableModel.load(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    @contextmanager
    def interventions_active(self, active_subspaces):
        """
        Context manager for handling interventions (active subspaces).
        Temporarily sets self._active_subspaces, then restores after.
        """
        old_subspaces = getattr(self, '_active_subspaces', None)
        self._active_subspaces = active_subspaces
        try:
            yield
        finally:
            self._active_subspaces = old_subspaces

    def forward(self, inputs=None, **kwargs):
        """
        Overridden forward that allows passing a single dictionary
        as the first positional argument.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids", None)
            attention_mask = inputs.get("attention_mask", None)
            labels = inputs.get("labels", None)
            leftover_kwargs = {k: v for k, v in inputs.items()
                               if k not in {"input_ids", "attention_mask", "labels"}}
            leftover_kwargs.update(kwargs)

            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **leftover_kwargs
            )
        return super().forward(inputs, **kwargs)

    def print_trainable_parameters(self):
        """
        Print trainable parameters.
        """
        _linked_key_set = set([])
        trainable_intervention_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v, pv.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        trainable_intervention_parameters += count_parameters(v)
                else:
                    trainable_intervention_parameters += count_parameters(v)

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        all_model_parameters = sum(
            p.numel() for p in self.model.parameters()
        )

        total_trainable_parameters = trainable_intervention_parameters + trainable_model_parameters

        print(
            f"trainable intervention params: {trainable_intervention_parameters:,d} || "
            f"trainable model params: {trainable_model_parameters:,d}\n"
            f"model params: {all_model_parameters:,d} || "
            f"trainable%: {100 * total_trainable_parameters / all_model_parameters:.2f}"
        )

    def generate(self, base=None, **kwargs):
        """
        Make `model.generate(base=..., **kwargs)` call the underlying HF model's generate().

        For example:
          outputs = model.generate(
              base={"input_ids": <tensor>, "attention_mask": <tensor>, ...},
              max_length=..., do_sample=..., etc.
          )
        """
        if base is None:
            return self.model.generate(**kwargs)

        generation_kwargs = dict(base)
        generation_kwargs.update(kwargs)

        
        return self.model.generate(**generation_kwargs)
