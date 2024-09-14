from typing import Dict
import re

class ModelStructure:
    PATTERN_PREFIX: str = ""
    LAYER_PATTERNS: Dict[str, str] = {}
    ATTENTION_LAYERS = ("query", "key", "value")
    FFN_LAYERS = ("interm_dense", "output_dense")

    @classmethod
    def get_module_intra_layer_position(cls, module_name):
        for i, pattern_info in enumerate(cls.LAYER_PATTERNS.items()):
            pattern_name, pattern = pattern_info
            if pattern in module_name:
                return i, pattern_name
        LAYER_PATTERNS_STRING = "\n".join([f"{k}: {v}" for k, v in cls.LAYER_PATTERNS.items()])
        raise RuntimeError(f"module name {module_name} does not match patterns {LAYER_PATTERNS_STRING}")

    @classmethod
    def is_attention(cls, module_name, exclude_att_dense=False):
        patterns = cls.ATTENTION_LAYERS if exclude_att_dense else cls.MHA_LAYERS
        for i, pattern in enumerate(patterns):
            if cls.LAYER_PATTERNS[pattern] in module_name:
                return True
        return False

    @classmethod
    def is_ffn(cls, module_name):
        if cls.is_attention(module_name, exclude_att_dense=False):
            return False
        for i, pattern in enumerate(cls.FFN_LAYERS):
            if cls.LAYER_PATTERNS[pattern] in module_name:
                return True
        return False

    @classmethod
    def get_position_ffn(cls, module_name):
        for i, pattern in enumerate(cls.FFN_LAYERS):
            if cls.LAYER_PATTERNS[pattern] in module_name:
                return i
        FFN_PATTERNS_STRING = ", ".join([f"{v}" for k, v in cls.LAYER_PATTERNS.items() if k in cls.FFN_LAYERS])
        raise RuntimeError(f"Module name {module_name} does not match any of the FFN patterns : {FFN_PATTERNS_STRING}")

    @classmethod
    def is_decoder(cls, module_name):
        return True if 'decoder' in module_name else False

    @classmethod
    def layer_index(cls, child_module_name):
        extracts = re.findall(r"[0-9]+", child_module_name)
        return int(extracts[0])

    @staticmethod
    def is_layernorm(module_name):
        return "layernorm" in module_name.lower().replace("_", "")