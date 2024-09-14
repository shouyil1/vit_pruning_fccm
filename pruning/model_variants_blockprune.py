from typing import Dict
import re
from transformers import BertConfig, BartConfig, T5Config, DeiTConfig
from nn_pruning.modules.model_structure_generic import ModelStructure

from deit_token_drop_variant_i import DeiTConfigDropTokens as DeiTConfigDropTokens_Variant_I


class BertStructure(ModelStructure):
    PATTERN_PREFIX = "bert.encoder.layer.[0-9]+."
    LAYER_PATTERNS = dict(
        query="attention.self.query",
        key="attention.self.key",
        value="attention.self.value",
        att_dense="attention.output.dense",
        interm_dense="intermediate.dense",
        output_dense="output.dense",
    )
    ATTENTION_PREFIX = ("attention.self",)
    ATTENTION_LAYERS = ("query", "key", "value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="hidden_size",
        intermediate_size="intermediate_size",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="attention_head_size",
    )


####### DP: added DeiT support ######################
class DeiTStructure(ModelStructure):
    PATTERN_PREFIX = "deit.encoder.layer.[0-9]+."
    LAYER_PATTERNS = dict(
        query="attention.attention.query",
        key="attention.attention.key",
        value="attention.attention.value",
        att_dense="attention.output.dense",
        interm_dense="intermediate.dense",
        output_dense="output.dense",
    )
    ATTENTION_PREFIX = ("attention.attention",)
    ATTENTION_LAYERS = ("query", "key", "value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="hidden_size",
        intermediate_size="intermediate_size",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="attention_head_size",
    )
#####################################################
    

######## DP: Note that DeiT with Drop Tokens has structure identical to DeiTStructure ########
##############################################################################################


class BartStructure(ModelStructure):
    PATTERN_PREFIX = "model.(en|de)coder.layers.[0-9]+."
    LAYER_PATTERNS = dict(
        query="self_attn.q_proj",
        key="self_attn.k_proj",
        value="self_attn.v_proj",
        att_dense="self_attn.out_proj",
        encoder_decoder_query="encoder_attn.q_proj",
        encoder_decoder_key="encoder_attn.k_proj",
        encoder_decoder_value="encoder_attn.v_proj",
        encoder_decoder_att_dense="encoder_attn.out_proj",
        interm_dense="fc1",
        output_dense="fc2",
    )
    ATTENTION_PREFIX = ("self_attn", "encoder_attn")
    ATTENTION_LAYERS = ("query", "key", "value", "encoder_decoder_query", "encoder_decoder_key", "encoder_decoder_value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense", "encoder_decoder_att_dense")
    NAME_CONFIG = dict(
        hidden_size="d_model",
        intermediate_size="encoder_ffn_dim",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_heads",
        attention_head_size = "head_dim",
    )

class T5Structure(ModelStructure):
    PATTERN_PREFIX = "(en|de)coder.block.[0-9]+.layer.[0-9]+."
    LAYER_PATTERNS = dict(
        query="SelfAttention.q",
        key="SelfAttention.k",
        value="SelfAttention.v",
        att_dense="SelfAttention.o",
        encoder_decoder_query="EncDecAttention.q",
        encoder_decoder_key="EncDecAttention.k",
        encoder_decoder_value="EncDecAttention.v",
        encoder_decoder_att_dense="EncDecAttention.o",
        interm_dense="DenseReluDense.wi",
        output_dense="DenseReluDense.wo",
    )
    ATTENTION_PREFIX = ("SelfAttention", "EncDecAttention")
    ATTENTION_LAYERS = ("query", "key", "value", "encoder_decoder_query", "encoder_decoder_key", "encoder_decoder_value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense", "encoder_decoder_att_dense")
    NAME_CONFIG = dict(
        hidden_size="d_model",
        intermediate_size="d_ff",
        num_hidden_layers="num_layers",
        num_attention_heads="num_heads",
        attention_head_size="key_value_proj_dim",
    )

config2struct = {
    BertConfig: BertStructure,
    BartConfig: BartStructure,
    T5Config: T5Structure,
    DeiTConfig: DeiTStructure,
    DeiTConfigDropTokens_Variant_I: DeiTStructure

}

name2struct = {
    "bert": BertStructure,
    "bart": BartStructure,
    "t5": T5Structure,
    "deit": DeiTStructure,
    "deitdroptokens": DeiTStructure
}

class ModelStructureNotFound(RuntimeError):
    pass

def struct_from_config(config):
    structure = None
    if hasattr(config, "config_class"):
        structure = config2struct.get(config.config_class)
    elif hasattr(config, "model_type"):
        structure = name2struct.get(config.model_type)

    if structure is None:
        raise ModelStructureNotFound(f"Model config does not match any of the defined structures.")
    return structure

def struct_from_name(model_name):
    for name in name2struct.keys():
        if name in model_name:
            return name2struct[name]
    raise ModelStructureNotFound(f"Model name {model_name} does not match any of the defined structures.")

def struct_from_model(model):
    for structure in config2struct.values():
        layer_pattern = structure.LAYER_PATTERNS
        num_pattern = len(layer_pattern)
        for pattern in layer_pattern.values():
            for k, v in model.items():
                if pattern in k:
                    num_pattern -= 1
                    break
        if num_pattern == 0:
            return structure
    else:
        raise RuntimeError("Model does not match any of the defined structures.")

def count_num_heads(model):
    head_count = 0
    model_structure = struct_from_config(model.config_class)
    for name, module in model.named_modules():
        for attention_prefix in model_structure.ATTENTION_PREFIX:
            if name.endswith(attention_prefix):
                if hasattr(module, 'num_attention_heads'):
                    num_attention_heads = module.num_attention_heads
                elif hasattr(module, 'num_heads'):
                    num_attention_heads = module.num_heads
                else:
                    raise RuntimeError(f"Not able to retrieve number of attention head")
                head_count += num_attention_heads
    return head_count

