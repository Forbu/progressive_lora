"""
The idea of the module is to train neural network in a new way :
using LoRa to to successive finetuning
"""

import torch
from torch import nn

from minlora import (
    LoRAParametrization,
    add_lora,
    apply_to_lora,
    merge_lora,
)

from functools import partial

# gpt2 lora config
lora_config = {
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


def create_lora(model, lora_config=lora_config):
    """
    Main function of the module
    """
    _ = torch.set_grad_enabled(False)

    # TODO : implement the function
    add_lora(model, lora_config=lora_config)
    model.apply(apply_to_lora(lambda x: nn.init.ones_(x.lora_B)))

    # for a specific layer named Head (nn.Module)
    # we don't want to apply LoRa
    merge_lora(model.Head)

    return model

def save_lora(model, path):
    """
    Save the lora parameters of the model
    """
    # merge lora parameters
    merge_lora(model)

    torch.save(model.state_dict(), path)







