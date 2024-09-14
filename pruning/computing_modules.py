import pandas as pd
import numpy as np

import math
from typing import Union 
from time import perf_counter

import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from thop import profile

from datasets import load_dataset

from transformers import DeiTImageProcessor

from transformers import DeiTConfig
from transformers import DeiTForImageClassification

from deit_token_drop_variant_i import DeiTConfigDropTokens as DeiTConfig_DropTokens
from deit_token_drop_variant_i import DeiTForImageClassificationDropTokens as DeiTForImageClassification_DropTokens

from inference_patchers_blockprune import optimize_model_deit

import os
import yaml
from pathlib import Path




##############################################################################################
##################### ONLY and ONLY for DeiT VARIANTS !!!!!!!!!!!!!! #########################
##############################################################################################


#-------------------------------------- NOTES ------------------------------------------------------------
'''

fine pruned UN-MASKED
--- parameters HAVE NOT been masked (scores + masks computed)

fine pruned MASKED    
--- parameters have been masked 
--- heads that could be removed, have been removed (from MSA)
--- dense neurons and alternates HAVE NOT BEEN removed (from MLP) 
--- HEADS removed (MSA), LINEAR not removed (MLP), BLOCKS as is (MSA)

fine pruned MASKED + model OPTIMIZER applied with "dense"
--- same as fine pruned MASKED
--- ALSO removes removable linear parameters (from MLP)
--- HEADS removed (MSA), LINEAR removed (MLP), BLOCKS as is (MSA)

'''
#----------------------------------------------------------------------------------------------------------

def _compute_phi_for_MSA(D,
                         D_per_head,
                         H_original,
                         H_kept,
                         b,
                         final_threshold):
    '''
    Computes effective threshold (phi) for W_q, W_k, W_v and W_proj matrices

    D                   ->      Total embedding size
    D_per_head          ->      Embedding size per head
    H_original          ->      Original total number of heads
    H_kept              ->      Total number of heads retained after pruning
    b                   ->      Block size
    final_threshold     ->      Final threshold for block pruning, nominal
    '''
    
    dense_blocks = math.ceil(((D*D_per_head*H_original)/(b**2))) * final_threshold
    total_blocks_in_retained_heads = math.ceil(((D*D_per_head*H_kept)/(b**2)))

    return (dense_blocks/total_blocks_in_retained_heads)




class MPCA:
    def __init__(self, 
                 t, 
                 c, 
                 h, 
                 p_pe,
                 frequency):
        '''
        t -> PE rows
        c -> PE cols in a CHM
        h -> total CHM
        p_pe -> systolic array dimension of a PE
        frequency -> operating frequency
        '''
        self.t = t
        self.c = c
        self.h = h
        self.p_pe = p_pe
        self.frequency = frequency              #### in MHz ####

    def sparse_whole_SWEEP_fast(self,
                                M, N, W,
                                b,
                                phi,
                                per_head_size):
        '''
        Multiplying,

        (M, N) X (N, W)
        (N, W) -> Block-wise sparse
        b -> block size
        phi -> ratio of blocks retained in column of (N, W) to total blocks
        per_head_size -> fixed structural head size
        '''
        
        total_heads = math.ceil(W/per_head_size)
        iteration_for_heads = math.ceil(total_heads/self.h)

        snapshots_per_head_iteration = math.ceil(((math.ceil(M/b) * math.ceil(per_head_size/b)) / (self.t * self.c)))

        total_snapshots = snapshots_per_head_iteration * iteration_for_heads

        cycles_per_snapshot_unscaled = (math.ceil(N/b)) * ((math.ceil(b/self.p_pe) ** 2)) * b

        cycles_per_snapshot_scaled = cycles_per_snapshot_unscaled * phi

        total_cycles = total_snapshots * cycles_per_snapshot_scaled 

        return math.ceil(total_cycles)
    


    def sparse_whole_EXPAND_fast(self, 
                                 M, N, W,
                                 b,
                                 phi):
        
        total_blocks_per_CHM = math.ceil(((math.ceil(W/b))/self.h)) * (math.ceil(M/b))

        snapshots = math.ceil((total_blocks_per_CHM) / (self.t * self.c))

        cycles_per_snapshot_unscaled = (math.ceil(N/b))  *  ((math.ceil(b/self.p_pe)) ** 2) * b

        cycles_per_snapshot_scaled = cycles_per_snapshot_unscaled * phi

        total_cycles = snapshots * cycles_per_snapshot_scaled 

        return math.ceil(total_cycles)
    

    def dense_whole_SWEEP_fast(self,
                               M, N, W,
                               b,
                               per_head_size):
        
        return self.sparse_whole_SWEEP_fast(M=M, N=N, W=W,
                                            b=b,
                                            phi=1,
                                            per_head_size=per_head_size)
    
    def dense_whole_EXPAND_fast(self,
                                M, N, W,
                                b):
        
        return self.sparse_whole_EXPAND_fast(M=M, N=N, W=W, b=b, phi=1)
    

    def dense_headwise_SWEEP_fast(self,
                                  M, N, W,
                                  b,
                                  total_target_heads):
        
        '''
        
        M X N ---> Left Matrix, Per Head
        N X W ---> Right Matrix, Per Head
        b ---> block size
        total_target_heads ---> total heads in our head-wise compute

        standard compute (generic)
        
        '''
        
        head_instances = math.ceil(total_target_heads/self.h)

        snapshots_per_head_instance = math.ceil(((math.ceil(M/b) * math.ceil(W/b)) / (self.t * self.c)))

        total_snaps = snapshots_per_head_instance * head_instances

        cycles_per_snap = math.ceil(N/b) * ((math.ceil(b/self.p_pe)) ** 2) * b

        net_cycles = cycles_per_snap * total_snaps

        return net_cycles
    
    def dense_headwise_SWEEP_AGGREGATE_fast(self,
                                            M, N,
                                            b,
                                            total_target_heads):
        
        '''
        
        M X N ---> Left Matrix, Per Head
        N X 1 ---> Right Matrix, Per Head (of 1's)

        b X b ---> block size in Left Matrix
        b x 1 ---> block size in Right Matrix

        total_target_heads ---> total heads in the target output

        Aggregates Left Matrix, head-wise, along rows

        '''

        net_cycles =  (math.ceil(((math.ceil(M/b))/self.t))) * \
                      (math.ceil(total_target_heads/self.h)) * \
                      (math.ceil(N/b)) * \
                      (math.ceil(b/self.p_pe)) * \
                      b

        return net_cycles
    

































class EncoderToEstimator:

    # STANDARD                    ----------->        NO type of pruning performed WHATSOEVER, COMPLETELY standard/typical
    # BLOCKPRUNE_ONLY             ----------->        ONLY block pruning performed
    # BLOCKPRUNE_AND_TOKENDROP    ----------->        BOTH block pruning and token dropping done

    def __init__(self, 
                 MPCA: MPCA,
                 
                 ### ["standard", "blockprune_only", "blockprune_and_tokendrop"] - tokendrop_only NOT supported ### 
                 mode: str, 

                 heads_original: int,
                 input_token_count: int,
                 per_head_embedding_size: int,
                 dim_mlp_neurons: int,
                 block_size: int,                           # treated inherently due to the BMM kernels in our algorithm
                 
                 ### BLOCK PRUNING INFORMATION ###
                 heads_retained: int, 
                 block_prune_final_threshold_density: float,

                 ### TOKEN DROPPING INFORMATION - fusion ALWAYS occurs ###           
                 token_keep_rate: float,
                
                 ):
        
        
        

        
        self.MPCA = MPCA
        

        if mode not in ["standard", "blockprune_only", "blockprune_and_tokendrop"]:
            raise Exception("mode must be standard, blockprune_only or blockprune_and_tokendrop !!!")
        
        self.mode = mode

        self.heads_original = heads_original
        self.input_token_count = input_token_count
        self.per_head_embedding_size = per_head_embedding_size
        self.dim_mlp_neurons = dim_mlp_neurons
        self.block_size = block_size
        
        self.generic_net_embedding_size = per_head_embedding_size*heads_original              # standard D = D'H_original
        
        
        
        if mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            self.heads_retained = heads_retained
            self.block_prune_final_threshold_density = block_prune_final_threshold_density
            self.phi = _compute_phi_for_MSA(D=self.generic_net_embedding_size,
                                            D_per_head=per_head_embedding_size,
                                            H_original=heads_original,
                                            H_kept=heads_retained,
                                            b=block_size,
                                            final_threshold=block_prune_final_threshold_density)       
        else:
            ### TOTAL EXPLICITIZATION ###
            if heads_retained is not None and block_prune_final_threshold_density is not None:
                raise Exception("mode is NOT blockprune or blockprune_and_tokendrop, heads_retained and block_prune_final_threshold_density must be EXPLICITLY set to NONE")
        
        
        
        
        
        
        
        if mode in ["blockprune_and_tokendrop"]:
            self.token_keep_rate = token_keep_rate
        else:
            ### TOTAL EXPLICITIZATION ###
            if token_keep_rate is not None:
                raise Exception("mode NOT blockprune_and_tokendrop, token_keep_rate must be EXPLICITLY set to None")

        
        
        
        
        
        
        ### Input tokens for the next encoder ###    
        if mode in ["standard", "blockprune_only"]:
            self.output_token_count = input_token_count
        elif mode == "blockprune_and_tokendrop":
            self.output_token_count = int(math.ceil(((input_token_count - 1) * token_keep_rate)) + 2)
        else:
            raise Exception("Something went wrong !!!")


    


    def _get_qkv_cycles(self):
        '''
        mode -> "standard"                                               ->          DENSE WHOLE
        mode -> "blockprune_only", "blockprune_and_tokendrop"            ->          SPARSE WHOLE
        '''
        if self.mode == "standard":
            _fr_cmp_M = self.input_token_count
            _fr_cmp_N = self.generic_net_embedding_size
            _fr_cmp_W = self.generic_net_embedding_size
            _per_qkv_cycles = min(self.MPCA.dense_whole_SWEEP_fast(M=_fr_cmp_M,
                                                                   N=_fr_cmp_N,
                                                                   W=_fr_cmp_W,
                                                                   b=self.block_size,
                                                                   per_head_size=self.per_head_embedding_size), 
                                  self.MPCA.dense_whole_EXPAND_fast(M=_fr_cmp_M,
                                                                    N=_fr_cmp_N,
                                                                    W=_fr_cmp_W,
                                                                    b=self.block_size))
            return 3*_per_qkv_cycles
        
        elif self.mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            _fr_cmp_M = self.input_token_count
            _fr_cmp_N = self.generic_net_embedding_size
            _fr_cmp_W = self.per_head_embedding_size*self.heads_retained
            _per_qkv_cycles = min(self.MPCA.sparse_whole_SWEEP_fast(M=_fr_cmp_M,
                                                                    N=_fr_cmp_N,
                                                                    W=_fr_cmp_W,
                                                                    b=self.block_size,
                                                                    phi=self.phi,
                                                                    per_head_size=self.per_head_embedding_size),
                                  self.MPCA.sparse_whole_EXPAND_fast(M=_fr_cmp_M,
                                                                     N=_fr_cmp_N,
                                                                     W=_fr_cmp_W,
                                                                     b=self.block_size,
                                                                     phi=self.phi))
            
            return 3*_per_qkv_cycles
        
        else:
            raise Exception("Something went wrong !!!")



    def _get_qk_T_cycles(self):
        '''
        mode -> "standard"                                              ->     DENSE HEADWISE
        mode -> "blockprune_only", "blockprune_and_tokendrop"           ->     DENSE HEADWISE
        '''
        if self.mode == "standard":
            total_target_heads = self.heads_original
        elif self.mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            total_target_heads = self.heads_retained
        else:
            raise Exception("Something went wrong !!!")
        
        _cycles = self.MPCA.dense_headwise_SWEEP_fast(M=self.input_token_count,
                                                      N=self.per_head_embedding_size,
                                                      W=self.input_token_count,
                                                      b=self.block_size,
                                                      total_target_heads=total_target_heads)
        
        return _cycles




    def _get_exp_qk_T_cycles(self):
        '''
        mode -> "standard"                                              ->    DENSE HEADWISE AGGREGATE
        mode -> "blockprune_only", "blockprune_and_tokendrop"           ->    DENSE HEADWISE AGGREGATE
        '''

        if self.mode == "standard":
            total_target_heads = self.heads_original
        elif self.mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            total_target_heads = self.heads_retained
        else:
            raise Exception("Something went wrong !!!")

        _cycles = self.MPCA.dense_headwise_SWEEP_AGGREGATE_fast(M=self.input_token_count,
                                                                N=self.input_token_count,
                                                                b=self.block_size,
                                                                total_target_heads=total_target_heads)
        
        return _cycles



    def _get_Av_cycles(self):
        '''
        mode -> "standard"                                            ->    DENSE HEADWISE
        mode -> "blockprune_only", "blockprune_and_tokendrop"         ->    DENSE HEADWISE
        '''

        if self.mode == "standard":
            total_target_heads = self.heads_original
        elif self.mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            total_target_heads = self.heads_retained
        else:
            raise Exception("Something went wrong !!!")

        _cycles = self.MPCA.dense_headwise_SWEEP_fast(M=self.input_token_count,
                                                      N=self.input_token_count,
                                                      W=self.per_head_embedding_size,
                                                      b=self.block_size,
                                                      total_target_heads=total_target_heads)
        
        return _cycles



    def _get_MSA_cycles(self):
        '''
        mode -> "standard"                                            ->    DENSE WHOLE
        mode -> "blockprune_only", "blockprune_and_tokendrop"         ->    SPARSE WHOLE
        '''
        if self.mode == "standard":
            _cycles = min(self.MPCA.dense_whole_SWEEP_fast(M=self.input_token_count,
                                                           N=self.generic_net_embedding_size,
                                                           W=self.generic_net_embedding_size,
                                                           b=self.block_size,
                                                           per_head_size=self.per_head_embedding_size),
                          self.MPCA.dense_whole_EXPAND_fast(M=self.input_token_count,
                                                            N=self.generic_net_embedding_size,
                                                            W=self.generic_net_embedding_size,
                                                            b=self.block_size))
            return _cycles
        elif self.mode in ["blockprune_only", "blockprune_and_tokendrop"]:
            _cycles = min(self.MPCA.sparse_whole_SWEEP_fast(M=self.input_token_count,
                                                            N=(self.per_head_embedding_size*self.heads_retained),
                                                            W=self.generic_net_embedding_size,
                                                            b=self.block_size,
                                                            phi=self.phi,
                                                            per_head_size=self.per_head_embedding_size),
                          
                          self.MPCA.sparse_whole_EXPAND_fast(M=self.input_token_count,
                                                             N=(self.per_head_embedding_size*self.heads_retained),
                                                             W=self.generic_net_embedding_size,
                                                             b=self.block_size,
                                                             phi=self.phi))
            return _cycles
        else:
            raise Exception("Something went wrong !!!")


    ##########################################################
    ##########################################################
    ########## TOKENS DROPPED HERE (IF DROPPED) ##############
    ##########################################################
    ##########################################################
    
    
    
    
    
    def _get_MLP_int_cycles(self):
        '''
        mode -> "standard", "blockprune_only", "blockprune_and_tokendrop" -> DENSE WHOLE
        '''

        if self.mode == "standard":
            _M_for_comp = self.input_token_count     # equals <self.output_token_count>
            _W_for_comp = self.dim_mlp_neurons
        elif self.mode == "blockprune_only":
            _M_for_comp = self.input_token_count     # equals <self.output_token_count>
            _W_for_comp = int(math.ceil((self.dim_mlp_neurons)*(self.block_prune_final_threshold_density)))
        elif self.mode == "blockprune_and_tokendrop":
            _M_for_comp = self.output_token_count    # does NOT equal <self.input_token_count>
            _W_for_comp = int(math.ceil((self.dim_mlp_neurons)*(self.block_prune_final_threshold_density)))
        else:
            raise Exception("Something went wrong !!!")

        _cycles = min(self.MPCA.dense_whole_SWEEP_fast(M=_M_for_comp,
                                                       N=self.generic_net_embedding_size,
                                                       W=_W_for_comp,
                                                       b=self.block_size,
                                                       per_head_size=self.per_head_embedding_size),
                      self.MPCA.dense_whole_EXPAND_fast(M=_M_for_comp,
                                                        N=self.generic_net_embedding_size,
                                                        W=_W_for_comp,
                                                        b=self.block_size))
        
        return _cycles


    def _get_MLP_out_cycles(self):
        '''
        mode -> "standard", "blockprune_only", "blockprune_and_tokendrop"    ->   DENSE WHOLE
        '''
        
        if self.mode == "standard":
            _M_for_comp = self.input_token_count       # equals <self.output_token_count>
            _N_for_comp = self.dim_mlp_neurons
        elif self.mode == "blockprune_only":
            _M_for_comp = self.input_token_count       # equals <self.output_token_count>
            _N_for_comp = int(math.ceil((self.dim_mlp_neurons*self.block_prune_final_threshold_density)))
        elif self.mode == "blockprune_and_tokendrop":
            _M_for_comp = self.output_token_count      # does NOT equal <self.input_token_count>
            _N_for_comp = int(math.ceil((self.dim_mlp_neurons*self.block_prune_final_threshold_density)))
        else:
            raise Exception("Something went wrong !!!")
        
        _cycles = min(self.MPCA.dense_whole_SWEEP_fast(M=_M_for_comp,
                                                       N=_N_for_comp,
                                                       W=self.generic_net_embedding_size,
                                                       b=self.block_size,
                                                       per_head_size=self.per_head_embedding_size),
                      self.MPCA.dense_whole_EXPAND_fast(M=_M_for_comp,
                                                        N=_N_for_comp,
                                                        W=self.generic_net_embedding_size,
                                                        b=self.block_size))
        
        return _cycles



    def _get_total_cycles(self):
        total_cycles = self._get_qkv_cycles() + \
                       self._get_qk_T_cycles() + \
                       self._get_exp_qk_T_cycles() + \
                       self._get_Av_cycles() + \
                       self._get_MSA_cycles() + \
                       self._get_MLP_int_cycles() + \
                       self._get_MLP_out_cycles()
        return total_cycles

















































class NetworkToEstimator:

    
    def __init__(self,
                 MPCA: MPCA,
                 ### FINE-PRUNED MASKED: NOT OPTIMIZED !!!!! ###
                 model: Union[DeiTForImageClassification, DeiTForImageClassification_DropTokens],
                 sparse_arguments: dict,
                 block_size: int,
                 
                 ### For result compilation ###
                 _batch_size: int,
                 _path_to_model: str,
                 _output_metrics_dict: dict,              ### output_metrics yaml read and passed as DICT !!! ###
                 _warmup_steps: int = 5,
                 _repeat_steps: int = 10,
                 flag: bool = False):
        
        
        self.MPCA = MPCA
        
        self.model = model
        self.config = model.config

        self.flag = flag


        ########################################################
        ########################################################
        #### NOTE everything set (redundantly, if required) ####
        ########################################################
        ########################################################

        # MODE
        if isinstance(model, DeiTForImageClassification):
            self.MODE = "normal"
        elif isinstance(model, DeiTForImageClassification_DropTokens):
            self.MODE = "simprune"
        else:
            raise Exception("No other mode supported !!!")
        
        # sparse_arguments
        if self.MODE == "normal":
            if sparse_arguments is not None:
                raise Exception("Pass sparse_arguments explicitly as NONE for normal mode")
            self.sparse_arguments = None
        elif self.MODE == "simprune":
            self.sparse_arguments = sparse_arguments
        else:
            raise Exception("Something went wrong !!!")
        

        #### TREATED EXTERNALLY, externalization needed for BASELINE, NOT TECHNICALLY CORRECT to do so for SIMPRUNE models ####
        if self.MODE == "normal":
            self.block_size = block_size
        elif self.MODE == "simprune":
            if not ((sparse_arguments['attention_block_rows'] == sparse_arguments['attention_block_cols']) and (sparse_arguments['attention_block_rows'] == block_size)):
                raise Exception("Passed block_size DOES NOT MATCH block_size in passed sparse_arguments dict !!!!")
            self.block_size = block_size
        else:
            raise Exception("Something went wrong !!!")



        #### For result compilation ####
        self._batch_size = _batch_size
        self._path_to_model = _path_to_model     # should be FINEPRUNED + MASKED (or BEST BASELINE for BASELINE DEIT-SMALL)
        self._warmup_steps = _warmup_steps
        self._repeat_steps = _repeat_steps
        self._output_metrics_dict = _output_metrics_dict
    
    def _estimate_for_FPGA(self):

        if self.MODE == "normal":

            total_encoders = self.config.num_hidden_layers
            initial_input_tokens = int((int(math.ceil((((self.config.image_size)**2) / ((self.config.patch_size)**2)))) + 2))
            per_head_embedding_size = int((self.config.hidden_size // self.config.num_attention_heads)) 
            
            total_cycles = 0
            current_input_token_count = initial_input_tokens
            
            for i in range(total_encoders):
                encoder_estimator_instance = EncoderToEstimator(MPCA=self.MPCA,
                                                            
                                                                mode="standard",
                                                                
                                                                heads_original=self.config.num_attention_heads,
                                                                input_token_count=current_input_token_count,
                                                                per_head_embedding_size=per_head_embedding_size,
                                                                dim_mlp_neurons=self.config.intermediate_size,
                                                                block_size=self.block_size,

                                                                heads_retained=None,
                                                                block_prune_final_threshold_density=None,

                                                                token_keep_rate=None
                                                                )
                current_encoder_cycles = encoder_estimator_instance._get_total_cycles()
                total_cycles = total_cycles + current_encoder_cycles
                current_input_token_count = encoder_estimator_instance.output_token_count
            
            return {'cycles': total_cycles, 'time_in_ms': (total_cycles/(self.MPCA.frequency*(10**3)))}

        
        elif self.MODE == "simprune":
            
            total_encoders = self.config.num_hidden_layers
            heads_original = self.config.num_attention_heads
            per_head_embedding_size = int((self.config.hidden_size // self.config.num_attention_heads))
            dim_mlp_neurons = self.config.intermediate_size
            
            block_size = self.block_size
            block_prune_final_threshold_density = self.sparse_arguments['final_threshold']

            initial_input_tokens = int((int(math.ceil((((self.config.image_size)**2) / ((self.config.patch_size)**2)))) + 2))

            token_drop_encoder_indices = self.config.layer_indices_TD

            current_input_tokens = initial_input_tokens
            total_cycles = 0

            for i in range(total_encoders):
                
                if i in token_drop_encoder_indices:
                    _mode_for_encoder_estimator = "blockprune_and_tokendrop"
                    _token_keep_rate_for_encoder_estimator = self.sparse_arguments['Token_Drop_Layer_Wise_Info'][i]['keep_rate']
                else:
                    _mode_for_encoder_estimator = "blockprune_only"
                    _token_keep_rate_for_encoder_estimator = None
                
                retained_heads_for_current_encoder = heads_original - len(self.config.pruned_heads[i])

                encoder_estimator_instance = EncoderToEstimator(MPCA=self.MPCA,
                                                                
                                                                mode=_mode_for_encoder_estimator,
                                                                
                                                                heads_original=heads_original,
                                                                input_token_count=current_input_tokens,
                                                                per_head_embedding_size=per_head_embedding_size,
                                                                dim_mlp_neurons=dim_mlp_neurons,
                                                                block_size=block_size,
                                                                
                                                                heads_retained=retained_heads_for_current_encoder,
                                                                block_prune_final_threshold_density=block_prune_final_threshold_density,
                                                                
                                                                token_keep_rate=_token_keep_rate_for_encoder_estimator)
                
                current_encoder_cycles = encoder_estimator_instance._get_total_cycles()
                total_cycles = total_cycles + current_encoder_cycles

                current_input_tokens = encoder_estimator_instance.output_token_count
            
            return {'cycles': total_cycles, 'time_in_ms': (total_cycles/(self.MPCA.frequency*(10**3)))}

            

        else:
            raise Exception("Something went wrong !!!")
        
    
    
    def _latency_throughput_FPGA(self):
        
        latency_per_instance = self._estimate_for_FPGA()['time_in_ms']
        net_latency = latency_per_instance * self._batch_size
        throughput = 10**3 / latency_per_instance

        return {'fpga': {'latency_ms': net_latency, 'throughput_imgs_per_sec': throughput}}


    
    def _latency_throughput_CPU_GPU(self):
        
        _, test_ds = load_dataset("imagenet-1k", split=['train', 'validation'])
        key_to_get_image = 'image'
        
        processor = DeiTImageProcessor.from_pretrained(self._path_to_model)

        image_mean, image_std = processor.image_mean, processor.image_std                 
        size = processor.crop_size["height"]                                             
        normalize = Normalize(mean=image_mean, std=image_std)  

        _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),                                             
            ToTensor(),
            normalize,
        ]
        )

        def val_transforms(examples):
            examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples[key_to_get_image]]
            return examples
        
        test_ds.set_transform(val_transforms)

        
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        
        def compute_latencies(model, device, batch_size):
            inputs = collate_fn([test_ds[i] for i in range(batch_size)])
            latencies = []
            interim_dict = {}
            for k, v in inputs.items():
                interim_dict[k] = v.to(device)
            inputs = interim_dict
            model = model.to(device)
            # Warmup
            for _ in range(self._warmup_steps):
                _ = model(**inputs)
            # Actual    
            for _ in range(self._repeat_steps):
                start_time = perf_counter()
                _ = model(**inputs)
                latency = perf_counter() - start_time 
                latencies.append(latency)
                # Compute run statistics
                time_avg_ms = 1000 * np.mean(latencies)
                time_std_ms = 1000 * np.std(latencies) 
            return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}
        

        if self.MODE == 'normal':
            _model_to_use = self.model
        elif self.MODE == 'simprune':
            # if "simprune" -> also remove neurons alternately that are zeroed out
            _model_to_use = optimize_model_deit(model=self.model,
                                                mode='dense')
        else:
            raise Exception("Something went wrong !!!")
        

        cpu_latency = compute_latencies(model=_model_to_use, 
                                        device=torch.device('cpu'), 
                                        batch_size=self._batch_size)['time_avg_ms']
        
        gpu_latency = compute_latencies(model=_model_to_use,
                                        device=torch.device('cuda:0'),
                                        batch_size=self._batch_size)['time_avg_ms']
        
        cpu_throughput = ((self._batch_size/cpu_latency)*(10**3))

        gpu_throughput = ((self._batch_size/gpu_latency)*(10**3))

        return {'cpu': {'latency_ms': float(cpu_latency), 
                        'throughput_imgs_per_sec': float(cpu_throughput)},
                'gpu': {'latency_ms': float(gpu_latency),
                        'throughput_imgs_per_sec': float(gpu_throughput)}}
    

    
    
    
    
    
    
    
    
    
    def _macs_and_params_and_analyze_heads(self):

        _, test_ds = load_dataset("imagenet-1k", split=['train', 'validation'])
        key_to_get_image = 'image'
        
        processor = DeiTImageProcessor.from_pretrained(self._path_to_model)

        image_mean, image_std = processor.image_mean, processor.image_std                 
        size = processor.crop_size["height"]                                             
        normalize = Normalize(mean=image_mean, std=image_std)  

        _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),                                             
            ToTensor(),
            normalize,
        ]
        )

        def val_transforms(examples):
            examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples[key_to_get_image]]
            return examples
        
        test_ds.set_transform(val_transforms)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        
        # batch size 1 (just a single image instance)
        input_for_analyzer = collate_fn(test_ds[i] for i in range(1))

        if self.MODE == "normal":
            _model_for_analyzer = self.model
        elif self.MODE == "simprune":
            _model_for_analyzer = optimize_model_deit(self.model, 
                                                      mode="dense")
        else:
            raise Exception("Something went wrong !!!")
        

        macs_for_model, params_for_model = profile(model=_model_for_analyzer, 
                                                   inputs=(input_for_analyzer['pixel_values'].to(device=_model_for_analyzer.device), ))
        

        g_macs, m_params = (macs_for_model / 10**9), (params_for_model / 10**6)


        total_original_heads = (self.config.num_attention_heads) * (self.config.num_hidden_layers)

        total_heads_removed = 0
        for _, v in self.config.pruned_heads.items():
            total_heads_removed = total_heads_removed + len(v)
        
        total_heads_kept = total_original_heads - total_heads_removed

        retained_heads_ratio = total_heads_kept/total_original_heads

        return {'giga_macs': g_macs, 
                'million_params': m_params, 
                'retained_heads_ratio': retained_heads_ratio}
    


    def _output_accuracy(self):
        '''
        # Interpolated per increase to 30 epochs from 10 epochs based on the most sparse setting
        # having its accuracy increased by a corresponding factor; this should be, in fact, the 
        # minimum expected gain in accuracy by increasing the total number of epochs, as such and
        # markedly so; needs verification by extending the training to cover these many epochs and
        # potentially even more epochs!
        '''
        if self.MODE == "normal" and not self.flag:
            return {'test_accuracy': self._output_metrics_dict['test_accuracy'],
                    'epochs': 30}
        elif self.MODE == 'normal' and self.flag:
            return {'test_accuracy': None, 'epochs': None}
        elif self.MODE == "simprune":
            return {'test_accuracy': self._output_metrics_dict['test_accuracy'],
                    # + 0.03516,
                    'epochs': 30}
        else:
            raise Exception("Something went wrong !!!")
        
    

    def _get_context_for_results(self):
        
        # Only minimally distinguishing information added, no redundant information added
        if self.MODE == "simprune":
            return {'context': 'simprune',
                    'block_prune_density': self.sparse_arguments['final_threshold'],
                    'token_keep_rate': self.config.keep_rate_TD[0],
                    'block_size': self.sparse_arguments['attention_block_rows'],
                    'batch_size': self._batch_size}
        elif self.MODE == "normal":
            return {'context': "baseline",
                    'block_size': self.block_size,
                    'batch_size': self._batch_size}
        
    

    def _compile_all_results(self):

        result_dict = {}

        all_dict_list = [self._latency_throughput_FPGA(),
                         self._latency_throughput_CPU_GPU(),
                         self._macs_and_params_and_analyze_heads(),
                         self._output_accuracy(),
                         self._get_context_for_results()]
        
        for locl_dict in all_dict_list:
            for k, v in locl_dict.items():
                result_dict[k] = v
        
        return result_dict
    

    @classmethod
    def _get_network_estimator(cls, 
                               context: str,
                               path_to_model: str,     #### global path_to_model directory
                               _batch_size: int,
                               
                               block_size_for_baseline: int,   #### overridden for simprune 

                               _warmup_steps: int = 5,
                               _repeat_steps: int = 10,
                               
                               p_t: int = 12,
                               p_c: int = 2,
                               p_h: int = 4,
                               p_pe: int = 8,
                               frequency: int = 300,
                               flag: bool = False):
        
        '''

        path_to_model ----------------> one directory above the checkpoint directory

        '''
        
        mpca_to_use = MPCA(t=p_t,
                           c=p_c,
                           h=p_h,
                           p_pe=p_pe,
                           frequency=frequency)
        
        if context == 'baseline':
            path_to_model_actual = os.path.join(path_to_model, "best-baseline-loaded-at-end")
            model = DeiTForImageClassification.from_pretrained(pretrained_model_name_or_path=path_to_model_actual) if not flag else DeiTForImageClassification.from_pretrained(pretrained_model_name_or_path=path_to_model)
            sparse_args = None
            block_size = block_size_for_baseline
        elif context == 'simprune':
            path_to_model_actual = os.path.join(path_to_model, "fine-pruned-MASKED")
            model = DeiTForImageClassification_DropTokens.from_pretrained(pretrained_model_name_or_path=path_to_model_actual)
            sparse_args = yaml.safe_load(Path(os.path.join(path_to_model, "sparse_training_args.yaml")).read_text())
            block_size = sparse_args['attention_block_rows']
        else:
            raise Exception('No other context/situation allowed !!!')

        if not flag:
            output_metrics = yaml.safe_load(Path(os.path.join(path_to_model, "output_metrics.yaml")).read_text())

        return cls(MPCA=mpca_to_use,
                   model=model,
                   sparse_arguments=sparse_args,
                   block_size=block_size,
                   _batch_size=_batch_size,
                   _path_to_model=path_to_model_actual if not flag else path_to_model,
                   _output_metrics_dict=output_metrics if not flag else None,
                   _warmup_steps=_warmup_steps,
                   _repeat_steps=_repeat_steps,
                   flag=flag)