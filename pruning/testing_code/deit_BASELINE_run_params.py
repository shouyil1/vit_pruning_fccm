from transformers import ViTForImageClassification
import os
import time
from deit_token_drop_variant_i import DropTokenLayerContextLayerwise, DropTokenLayerContextCommon

timestr = time.strftime("%Y%m%d-%H%M%S")

#--------------------------------- DATASET AND MODEL -----------------------------------------------------------------------

# if HF access token not added, set to False
hf_access_token_for_imagenet_added = True                                         ###### <SET> #############  

dataset_choice_list_for_experiment = ["cifar-10", "imagenet-1k"]
model_choice_list_for_experiment = ["deit", "deit_drop_tokens_variant_i"]


dataset_to_utilize_for_experiment = "imagenet-1k"                                 ###### <SET> ###########
model_to_utilize_for_experiment = "deit_drop_tokens_variant_i"                    ###### <SET> ###########

if dataset_to_utilize_for_experiment not in dataset_choice_list_for_experiment:
    raise Exception("Dataset not legal for experiments!")
if model_to_utilize_for_experiment not in model_choice_list_for_experiment:
    raise Exception("Model type not legal for experiments!")

dataset_model_name_prefix = str(dataset_to_utilize_for_experiment) + str("_") + str(model_to_utilize_for_experiment)


#------------------------------- FINE-TUNED TEACHER TYPE AND LOCATION ------------------------------------------------------

# DATASET (class, label) consistency between TEACHER and STUDENT presumed (for SAME dataset utilized for training/fine-tuning teacher and fine-pruning student, this should be always TRUE unless corrupt data across instances of experimentation)
teacher_for_cifar_10 = "trained-models/inital-exp-baseline-std-vit-cifar-10/checkpoint-3375"        ######## <SET> #######
teacher_for_imagenet_1k = "google/vit-base-patch16-224"                                             ######## <SET> #######

distil_teacher_name_or_path_mapping_dict = {"cifar-10": teacher_for_cifar_10,
                                            "imagenet-1k": teacher_for_imagenet_1k}       

# Teacher constructor presumed to be default (ViTForImageClassification)
teacher_constructor = ViTForImageClassification
distil_teacher_name_or_path = distil_teacher_name_or_path_mapping_dict[dataset_to_utilize_for_experiment]



#------------------------------ TRAIN ARGS -------------------------------------------------------------------

num_gpus_used = 4                                                                           #### <SET> ####
common_train_eval_batch_size_per_device = 32                                                #### <SET> ####
learning_rate = 2e-5
total_epochs = 11                                                                            #### <SET> ####
weight_decay = 0.01
logging_steps = 1000                                                                         #### <SET> ####
warmup_steps = 3000                                                                           #### <SET> ####

#------------------------------- SPARSE ARGS -----------------------------------------------------------------

mask_scores_learning_rate = 1e-2

distil_alpha_ce = 0.1
distil_alpha_teacher = 0.9
distil_temperature = None

dense_pruning_method = "topK:1d_alt"                                                        #### <SET> ####
attention_pruning_method = "topK"                                                           #### <SET> ####
initial_threshold = 1.0                                                                     #### <SET> ####
final_threshold = 0.5                                                                      #### <SET> ####

attention_block_rows = 16                                                                   #### <SET> ####
attention_block_cols = 16                                                                   #### <SET> ####
dense_block_rows = None                                                                     #### <SET> ####
dense_block_cols = None                                                                     #### <SET> ####       

initial_warmup = 1                                                                          #### <SET> ####
final_warmup = 6                                                                            #### <SET> #### 

regularization = "disabled"
regularization_final_lambda = None
attention_lambda = None
dense_lambda = None








#################### MODEL SIZE if DeiT model type ##############################################################

if model_to_utilize_for_experiment in ["deit", "deit_drop_tokens_variant_i"]:
    # this should be "tiny", "small", "base"
    deit_model_size_to_use_for_experiment = "small"                                         ##### <SET> #####
    dataset_model_name_prefix = str(dataset_model_name_prefix) + str("_") + str(deit_model_size_to_use_for_experiment)
else:
    raise Exception("No other model size currently supported!")

#################################################################################################################








####################### MODEL TYPE SPECIFICS ####################################################################


#------------------------- DeiT Drop Token Variant I ---------------------------------------------------------------------

if model_to_utilize_for_experiment == "deit_drop_tokens_variant_i":
    ################ NOTE this is the only format explicity supported ############################################
    ################ NOTE KEYS -> Layer INDICES to Token Drop (NOT layer number with natural 1-index) ############
    ## INDICES NOT 1-INDEX-NATURAL ##
    info_dict_for_token_dropping = {                                                                   ##### <SET> if deit_drop_tokens_variant_i ######
                                        2:  {'keep_rate': 0.7,
                                            'fuse_inattentive': True},   #### KEY/INDEX -> i ------- LAYER/HUMAN-INDEX -> i + 1
                                        
                                        6:  {'keep_rate': 0.7,
                                            'fuse_inattentive': True},
                                        
                                        9:  {'keep_rate': 0.7,
                                            'fuse_inattentive': True}
                                    }
    token_drop_context_for_experiment = DropTokenLayerContextLayerwise(info_dict=info_dict_for_token_dropping)
elif model_to_utilize_for_experiment == "deit":
    info_dict_for_token_dropping = None
    token_drop_context_for_experiment = None
else: 
    raise Exception("No other model type supported!")

###################################################################################################################



#--------------------------- DeiT/DeiT Drop Token Variant I, force loading classifier weights -----------------------------

# allows loading of classifier params for imagenet despite the fine-pruning to reduce epochs

if dataset_to_utilize_for_experiment == "imagenet-1k" and (model_to_utilize_for_experiment == 'deit' or model_to_utilize_for_experiment == 'deit_drop_tokens_variant_i'):
    force_load_classifier_parameters_for_experiments = True           # <SET> #      (BOOL T/F)
else:
    force_load_classifier_parameters_for_experiments = None


#--------------------------------------------------------------------------------------------------------------------------





################### DIR name context for token dropping ###########################################################


if model_to_utilize_for_experiment == "deit_drop_tokens_variant_i":
    
    total_layers_for_token_dropping = len(list(info_dict_for_token_dropping.keys()))
    if total_layers_for_token_dropping == 0:
        raise Exception("With deit_token_drop_variant_i model, the total token dropping layers must be > 0")
    layer_type_context_for_token_dropping = "default" if (list(info_dict_for_token_dropping.keys()) == [2, 6, 9]) else "non-default"
    all_token_dropping_keep_rates = [x['keep_rate'] for x in list(info_dict_for_token_dropping.values())]
    keep_rate_context_for_token_dropping = all_token_dropping_keep_rates[0] if (len(set(all_token_dropping_keep_rates)) == 1) else "varying"
    all_token_dropping_fuse_inattentive = [x['fuse_inattentive'] for x in list(info_dict_for_token_dropping.values())]
    fuse_inattentive_context_for_token_dropping = all_token_dropping_fuse_inattentive[0] if (len(set(all_token_dropping_fuse_inattentive)) == 1) else "varying"

    net_info_word_for_token_dropping =      str("tokenDropInfo") + str("_") \
                                        +   str("layerCount") + str("_") \
                                        +   str(total_layers_for_token_dropping) + str("_") \
                                        +   str("layerType") + str("_") \
                                        +   str(layer_type_context_for_token_dropping) + str("_") \
                                        +   str("keepRate") + str("_") \
                                        +   str(keep_rate_context_for_token_dropping) + str("_") \
                                        +   str("fused") + str("_") \
                                        +   str(fuse_inattentive_context_for_token_dropping)

elif model_to_utilize_for_experiment == "deit":

    net_info_word_for_token_dropping =      str("tokenDropInfo_notApplicable")

else:

    raise Exception("No other model type supported!")















#---------------------------------- PATH ----------------------------------------------------------------------

# Adding some context to output dump
output_directory =   str("trained-models") + str("/") \
                   + str(dataset_model_name_prefix) + str("/") \
                   + str("epochs_") + str(total_epochs) \
                   + str("_blockPruningInfo") \
                   + str("_finalThreshold_") + str(final_threshold) \
                   + str("_blockSize_") + str(attention_block_rows) \
                   + str("_method_") + str(attention_pruning_method) \
                   + str("_") + str(net_info_word_for_token_dropping) \
                   + str("_") + str(timestr)                                                 



#---------------------------------------------------------------------------------------------------------------


#---------------------------------- RESUMING FROM CHECKPOINT --------------------------------------------------------


resume_training_for_a_variant_from_checkpoint = False                                                                    # <SET> #                           

# PASS output_directory as directory containing checkpoints and other file dumps
output_directory_for_a_variant_to_resume_training = None                                                                 # <SET> #

if resume_training_for_a_variant_from_checkpoint:
    print("NOTE: Training will be RESUMED from a checkpoint")
    output_directory = output_directory_for_a_variant_to_resume_training

#---------------------------------------------------------------------------------------------------------------------


#-------------------------------- PATHS for DUMPS -------------------------------------------------------------------------------

patched_model_save_location = os.path.join(output_directory, "patched_model")
save_model_fine_pruned_NOT_MASKED = os.path.join(output_directory, "fine-pruned-UNMASKED")
save_model_fine_pruned_MASKED = os.path.join(output_directory, "fine-pruned-MASKED")
cache_dir = os.path.join(output_directory, "model_cache_from_mpc")
logging_directory = os.path.join(output_directory, "logs")
output_metric_file = os.path.join(output_directory, "output_metrics.yaml")
normal_args_file = os.path.join(output_directory, "typical_training_args.yaml")
sparse_args_file = os.path.join(output_directory, "sparse_training_args.yaml")

#--------------------------------------------------------------------------------------------------------------------------------







#-------------------------- NOTE TOKEN DROPPING DEFAULT IF APPLICABLE -----------------------------------------------------------
'''

info_dict_for_token_dropping = {                                                                
                                        2:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True},   

                                        6:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True},
                                        
                                        9:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True}
                                    }

The above is standard default with tokens being dropped at,

Layer Indices -> 2, 6, 9

The actual layers (human-index) are,

Layer Human-Indices -> 3, 7, 10

PAPER suggests 3, 6, 9; we choose 2 instead of 3 to gain speed improvements (by early token dropping to reduce MACS further)

'''
#--------------------------------------------------------------------------------------------------------------------------------






if __name__ == "__main__":
    print(net_info_word_for_token_dropping)
    print(output_directory)