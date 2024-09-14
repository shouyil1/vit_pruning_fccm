import pandas as pd
import numpy as np
import math


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
        










class NetworkToEstimator:

    def __init__(self):
        pass