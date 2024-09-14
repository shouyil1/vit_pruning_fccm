import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers import DeiTConfig

from collections import OrderedDict
from typing import Mapping
from packaging import version


'''

Variant I (variant_i post fix)

Details below

NOTE additional variants may or may not be explored

'''

#################################################################
####### IMPORTANT TOKEN DROPPING NOTE ###########################
#################################################################
'''

Since our method modifies H -> H_kept,
    And we prune heads,
    Attention compute and token dropping scale affected,
    IMPORTANT NOTE: Examine and "secure" code if needed

'''

#################################################################
####### IMPORTANT DISTILLATION TOKEN NOTE #######################
'''

Note that this module,

                        - ALLOWS the distillation token to be dropped during token dropping (treats it as a standard token instead of a parameter token)

There are three variants at a high level,

VARIANT I   ----------->    Treat distillation token as a standard token,    
                            allow it to be dropped,           
                            don't use it for distillation (or for loss compute)

VARIANT II  ----------->    Treat distillation token as a special token,     
                            NEVER allow it to be dropped,     
                            don't use it for distillation (or for loss compute)

VARIANT III ----------->    Treat distillation token as a special token,     
                            NEVER allow it to be dropped,     
                            USE it for distillation (for loss compute)

VARIANT IV  ----------->    Treat distillation token as a special token,     
                            NEVER allow it to be dropped,     
                            USE it for distillation (for loss compute),                     
                            REFINE token dropping module with CLS + DSTL tokens

This code is for VARIANT I, simplest variant ------------> NOTE

'''




############# HELPER ########################
#############################################
def residual_indices(index, innermost_size):
    '''
    index, innermost_size:
            Indexing tensor of shape            [N_1, N_2, ...., N_k]
            Used to index a tensor of shape     [N_1, N_2, ...., innermost_size]
            N_k indices "selected" from possible innermost indices [0, 1, 2, ...., innermost_size - 1]

    returns:
            All innermost indices NOT in the N_k indices at each "location"
            Shape -> [N_1, N_2, ...., innermost_size - N_k]

    NOTE sanity checks not done
    NOTE referenced from youweiliang (https://github.com/youweiliang/evit/blob/97e58f610c51d4b74a070341739e41647dced32c/helpers.py#L52)
    '''
    all_index_tensor = torch.arange(0, innermost_size, device=index.device)
    _expanding_shape = index.shape[:-1] + (innermost_size, )
    all_index_tensor = all_index_tensor.expand(size=_expanding_shape)
    used_indices_masked = torch.scatter(all_index_tensor, 
                                        dim=-1, 
                                        index=index,
                                        value=0)
    residuals, _ = torch.sort(used_indices_masked, 
                              dim=-1,
                              descending=False)
    residuals = residuals[..., index.shape[-1]:].contiguous()
    return residuals



class DropTokenLayerContextCommon:
    
    def __init__(self,
                 keep_rate_common: float = 0.5,
                 fuse_inattentive_common: bool = True,
                 layer_indices: list[int] = [1, 4, 7]):
        
        self.keep_rate_common = keep_rate_common
        self.fuse_inattentive_common = fuse_inattentive_common
        self.layer_indices = layer_indices

    def process_common(self):

        layer_indices = self.layer_indices
        keep_rate_list = [self.keep_rate_common for i in range(len(layer_indices))]
        fuse_inattentive_list = [self.fuse_inattentive_common for i in range(len(layer_indices))]

        return layer_indices, keep_rate_list, fuse_inattentive_list

default_info_dict = {   
                        1:  {'keep_rate': 0.5,
                             'fuse_inattentive': True},
                         
                        4:  {'keep_rate': 0.5,
                             'fuse_inattentive': True},
                        
                        7:  {'keep_rate': 0.5,
                             'fuse_inattentive': True}
                    }

class DropTokenLayerContextLayerwise:

    def __init__(self,
                 info_dict: dict[int, dict[str, Union[float, bool]]] = default_info_dict):
        self.info_dict = info_dict

    def process_info_dict(self):
        
        layer_indices = []
        keep_rate_list = []
        fuse_inattentive_list = []

        for k, v in self.info_dict.items():
            layer_indices.append(k)
            keep_rate_list.append(v['keep_rate'])
            fuse_inattentive_list.append(v['fuse_inattentive'])

        return layer_indices, keep_rate_list, fuse_inattentive_list




# To be used with a DeiT encoder designed to drop tokens within specific sub-encoder layers
class DeiTConfigDropTokens(DeiTConfig):
    
    def __init__(self, 
                 token_dropping_info: Union[DropTokenLayerContextCommon, DropTokenLayerContextLayerwise] = DropTokenLayerContextLayerwise(),
                 **kwargs):
        
        if isinstance(token_dropping_info, DropTokenLayerContextCommon):
            self.layer_indices_TD, self.keep_rate_TD, self.fuse_inattentive_TD = \
            token_dropping_info.process_common() 
        elif isinstance(token_dropping_info, DropTokenLayerContextLayerwise):
            self.layer_indices_TD, self.keep_rate_TD, self.fuse_inattentive_TD = \
            token_dropping_info.process_info_dict()
        else:
            raise Exception("Invalid type for token_dropping_info!")

        super().__init__(**kwargs)

        # NOTE intense checks on types not done, assumed user provides sane floats for keep_rate and boolean flags for fuse_inattentive
        # If above not done, at a fine-grained stage, down the line, an error will be thrown due to checks at those levels

        # ONLY checking if maximum layer index provided is within bounds

        if not ((max(self.layer_indices_TD) <= (self.num_hidden_layers - 1)) and (min(self.layer_indices_TD) >= 0)):
            raise Exception("Provided layer indices are out of bounds of the total hidden layers present!")

        # Checking duplicates
        if len(set(self.layer_indices_TD)) != len(self.layer_indices_TD):
            raise Exception("Duplicates NOT allowed! Explicitly pass a list-set!")






class DeiTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = DeiTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_length, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)        ## DP: ?? ##
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask      ## DP: ??: mask -> 1, replace by mask_tokens, mask -> 0, keep original embedding ##

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings           ## DP: Automatically applied batch-wise, as such ##
        embeddings = self.dropout(embeddings)                        ## DP: Should do dropout independently across instances in a batch ##
        return embeddings








class DeiTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        
        ## DP: Learned token mapping? ##
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2) ## DP: N, C, H, W -> N, H*W, C
        return x












class DeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))          ## DP: Output -> B = batch, H = head, N = token, D' = attention hidden ##
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  ## DP: Should be B X H X N X N; multiplied element wise along outermost dimension elements ## 

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask    ## DP: Non-pruned mask; important to note this ##

        context_layer = torch.matmul(attention_probs, value_layer)  ## DP: Output = B X H X N X D' (elementwise)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  ## DP: B X N X H X D'
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape) ## DP: B X N X E 

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs






# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->DeiT
class DeiTSelfAttentionDropTokens(nn.Module):
    def __init__(self, config: DeiTConfig,
                 token_keep_rate: float, fuse_inattentive_tokens: bool = True, generate_warnings: bool = False) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if token_keep_rate == 0:
            raise Exception("Token keep rate cannot be 0!")
        
        if token_keep_rate == 1:
            raise Exception("Token keep rate is 1, please utilize DeiTSelfAttention default instead of DeiTSelfAttentionDropTokens")
        
        if not (token_keep_rate > 0 and token_keep_rate < 1):
            raise Exception("Token keep rate must be in the range (0, 1)")
        
        self.token_keep_rate = token_keep_rate

        self.fuse_inattentive_tokens = fuse_inattentive_tokens

        self.generate_warnings = generate_warnings

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        _, net_tokens, _ = hidden_states.shape            # B, N, E

        if not (net_tokens - 1 >= 1):
            raise Exception("Weird case! net_tokens less than 2 not supported!")

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))          ## DP: Output -> B = batch, H = head, N = token, D' = attention hidden ##
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  ## DP: Should be B X H X N X N; multiplied element wise along outermost dimension elements ## 

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask    ## DP: Non-pruned mask; important to note this ##

        cls_attention_scores = attention_probs[:, :, 0, 1:]  ## B X H X (N - 1)
        cls_attention_scores = cls_attention_scores.mean(dim=1) ## B X (N - 1) (aggregating along heads)
        
        total_keep_tokens = math.ceil(self.token_keep_rate * (net_tokens - 1))      # (net_tokens - 1) >= 1 and token_keep_rate is in (0, 1) --------> total_keep_tokens is in [1, 2, ..., (net_tokens - 1)]

        all_tokens_kept_flag = True if (total_keep_tokens == net_tokens - 1) else False

        if not all_tokens_kept_flag:
            _, top_k_indices = torch.topk(cls_attention_scores, k=total_keep_tokens, dim=1)
            non_top_k_indices = residual_indices(index=top_k_indices, 
                                                 innermost_size=cls_attention_scores.shape[-1])    
        else:
            top_k_indices, non_top_k_indices = None, None

        context_layer = torch.matmul(attention_probs, value_layer)  ## DP: Output = B X H X N X D' (elementwise)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  ## DP: B X N X H X D'
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape) ## DP: B X N X E 

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, cls_attention_scores, top_k_indices, non_top_k_indices, all_tokens_kept_flag
    









class DeiTSelfOutput(nn.Module):
    """
    The residual connection is defined in DeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    








class DeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.attention = DeiTSelfAttention(config)
        self.output = DeiTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        #### DP : ?? : Go through internal code in detail ####
        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)  ## DP: Note the dim=1 parameter here ##

        #### DP: ?? : Go through internal code in detail ####
        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs   ### DP: tuple(output, attention_scores) ###










class DeiTAttentionDropTokens(nn.Module):
    def __init__(self, config: DeiTConfig,
                 token_keep_rate: float, fuse_inattentive_tokens: bool = True, generate_warnings: bool = False) -> None:
        super().__init__()
        self.attention = DeiTSelfAttentionDropTokens(config, 
                                                     token_keep_rate=token_keep_rate,
                                                     fuse_inattentive_tokens=fuse_inattentive_tokens,
                                                     generate_warnings=generate_warnings)
        self.output = DeiTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        #### DP : ?? : Go through internal code in detail ####
        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)  ## DP: Note the dim=1 parameter here ##

        #### DP: ?? : Go through internal code in detail ####
        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs, cls_attention_scores, top_k_indices, non_top_k_indices, all_tokens_kept_flag = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, cls_attention_scores, top_k_indices, non_top_k_indices, all_tokens_kept_flag
    







class DeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states










class DeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states
    












class DeiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DeiTAttention(config)
        self.intermediate = DeiTIntermediate(config)
        self.output = DeiTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in DeiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in DeiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs













class DeiTLayerDropTokens(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig,
                 token_keep_rate: float, fuse_inattentive_tokens: bool = True, generate_warnings: bool = False) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DeiTAttentionDropTokens(config, 
                                                 token_keep_rate=token_keep_rate, fuse_inattentive_tokens=fuse_inattentive_tokens, generate_warnings=generate_warnings)
        self.intermediate = DeiTIntermediate(config)
        self.output = DeiTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.fuse_inattentive_tokens = fuse_inattentive_tokens
        self.token_keep_rate = token_keep_rate
        self.generate_warnings = generate_warnings

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        self_attention_outputs, cls_attention_scores, top_k_indices, non_top_k_indices, all_tokens_kept_flag  = self.attention(self.layernorm_before(hidden_states),  # in DeiT, layernorm is applied before self-attention
                                                                                                                               head_mask,
                                                                                                                               output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        ####################################################################
        ######## Drop Tokens in hidden_states ----> hidden_states ##########

        if not all_tokens_kept_flag:
            
            cls_hidden_states, remaining_hidden_states = hidden_states[:, 0, :], hidden_states[:, 1:, :] # cls_hidden_states -> B X E (class tokens), remaining_hidden_states -> B X (N - 1) X E (remaining tokens)
            cls_hidden_states = cls_hidden_states.unsqueeze(1)   # B X 1 X E
            top_k_indices = top_k_indices.unsqueeze(-1)            # B X topK X 1              
            _expanding_shape_top_k = top_k_indices.shape[:-1] + (remaining_hidden_states.shape[-1], )       # B X topK X E   
            top_k_indices = top_k_indices.expand(_expanding_shape_top_k)     # B X topK X E
            top_k_tokens_selected = torch.gather(input=remaining_hidden_states,
                                                 dim=1,
                                                 index=top_k_indices)        # B X topK X E (selected topK tokens)
            
            if self.fuse_inattentive_tokens:
                
                non_top_k_attention_scores = torch.gather(input=cls_attention_scores, 
                                                          dim=1, 
                                                          index=non_top_k_indices)      # B X nontopK
                non_top_k_indices = non_top_k_indices.unsqueeze(-1)     # B X nontopK X 1
                _expanding_shape_non_top_k = non_top_k_indices.shape[:-1] + (remaining_hidden_states.shape[-1], )    # B X nontopK X E
                non_top_k_indices = non_top_k_indices.expand(_expanding_shape_non_top_k)     # B X nontopK X E    
                non_top_k_tokens_selected = torch.gather(input=remaining_hidden_states,
                                                         dim=1,
                                                         index=non_top_k_indices)      # B X nontopK X E (inattentive tokens to fuse)
                non_top_k_attention_scores = non_top_k_attention_scores.unsqueeze(-1).permute(0, 2, 1)  # B X 1 X nontopK
                fused_tokens = torch.matmul(non_top_k_attention_scores, non_top_k_tokens_selected)   # B X 1 X E
                
                hidden_states = torch.cat([cls_hidden_states, top_k_tokens_selected, fused_tokens], dim=1) # B X (1 + topK + 1) X E
            
            else:

                hidden_states = torch.cat([cls_hidden_states, top_k_tokens_selected], dim=1)  # B X (1 + topK) X E      (inattentive tokens trashed)
        
        # else:
        #       hidden_states -> unchanged as all_tokens_kept_flag is True (all tokens kept, none dropped)




        ####################################################################
        # NOTE: Compute now on hidden_states with tokens dropped ###########
        ####################################################################

        
        # in DeiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs









##########################################################
##### Encoder stack for DeiT #############################
##########################################################
##########################################################
##### Standard DeiT layers + Drop Token DeiT layers ######
##########################################################

class DeiTEncoderDropTokens(nn.Module):
    def __init__(self, config: DeiTConfigDropTokens) -> None:
        super().__init__()
        self.config = config
        token_drop_info_dict = {key_index: [config.keep_rate_TD[i], config.fuse_inattentive_TD[i]] for i, key_index in enumerate(config.layer_indices_TD)}
        # Contains DeiTLayerDropTokens and DeiTLayer both
        layer_list = []
        for layer_index in range(config.num_hidden_layers):
            if layer_index in config.layer_indices_TD:
                layer_list.append(DeiTLayerDropTokens(config=config, 
                                                      token_keep_rate=token_drop_info_dict[layer_index][0], 
                                                      fuse_inattentive_tokens=token_drop_info_dict[layer_index][1],
                                                      generate_warnings=False))
            else: # layer_index not in config.layer_indices_TD -> layer with that layer_index is standard DeiT layer
                layer_list.append(DeiTLayer(config=config))
        # print(layer_list)
        self.layer = nn.ModuleList(layer_list)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    












# NOTE completely un-modified since does not need any token dropping context, as such (won't actually even be utilized)
class DeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

















class DeiTPreTrainedModelDropTokens(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # DP: modified
    config_class = DeiTConfigDropTokens

    # DP: modified ----------> unmodified to "deit"; refers to the base_model name in classification module which for us is self.deit -> DeiTModelDropTokens()
    base_model_prefix = "deit"
    
    main_input_name = "pixel_values"
    
    supports_gradient_checkpointing = True

    # DP: modified (still don't know what this is!)
    _no_split_modules = ["DeiTLayer", "DeiTLayerDropTokens"]      ### NOTE: Don't quite know what this is or what it does! ###
    # _no_split_moduels has some relation to mapping stuff in a module (sub-modules) to different devices, does not seem relevant for us


    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


































class DeiTModelDropTokens(DeiTPreTrainedModelDropTokens):
    def __init__(self, config: DeiTConfigDropTokens, add_pooling_layer: bool = True, use_mask_token: bool = False) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token)
        
        self.encoder = DeiTEncoderDropTokens(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.pooler = DeiTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    



















class DeiTForImageClassificationDropTokens(DeiTPreTrainedModelDropTokens):
    def __init__(self, config: DeiTConfigDropTokens) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.deit = DeiTModelDropTokens(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])      # [CLS] token only
        # we don't use the distillation token
        # DP: the distillation token can even be dropped, in general and will be treated as a standard token

        ###### DP: go through this code and elaborate ########################################
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        ###### DP: go through this code and elaborate, what is BCEWithLogitsLoss()? #########

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,                  # B X no. of labels (pre-softmax)
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
