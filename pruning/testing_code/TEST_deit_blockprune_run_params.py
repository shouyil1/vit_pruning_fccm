from transformers import ViTForImageClassification
import os
import time
from deit_token_drop_variant_i import DropTokenLayerContextLayerwise, DropTokenLayerContextCommon

timestr = time.strftime("%Y%m%d-%H%M%S")

#--------------------------------- DATASET AND MODEL -----------------------------------------------------------------------

dataset_choice_list_for_experiment = ["cifar-10", "imagenet-1k"]
model_choice_list_for_experiment = ["deit", "deit_drop_tokens_variant_i"]


dataset_to_utilize_for_experiment = "cifar-10"                                    ###### <SET> ###########
model_to_utilize_for_experiment = "deit_drop_tokens_variant_i"                    ###### <SET> ###########

if dataset_to_utilize_for_experiment not in dataset_choice_list_for_experiment:
    raise Exception("Dataset not legal for experiments!")
if model_to_utilize_for_experiment not in model_choice_list_for_experiment:
    raise Exception("Model type not legal for experiments!")

dataset_model_name_prefix = str(dataset_to_utilize_for_experiment) + str("_") + str(model_to_utilize_for_experiment)


#------------------------------- FINE-TUNED TEACHER TYPE AND LOCATION ------------------------------------------------------

distil_teacher_name_or_path_mapping_dict = {"cifar-10": "trained-models/inital-exp-baseline-std-vit-cifar-10/checkpoint-3375",
                                            "imagenet-1k": "TO-BE-DETERMINED"}

# Teacher constructor presumed to be default (ViTForImageClassification)
teacher_constructor = ViTForImageClassification
distil_teacher_name_or_path = distil_teacher_name_or_path_mapping_dict[dataset_to_utilize_for_experiment]



#------------------------------ TRAIN ARGS -------------------------------------------------------------------

num_gpus_used = 3                                                                           #### <SET> ####
common_train_eval_batch_size_per_device = 20                                                #### <SET> ####
learning_rate = 2e-5
total_epochs = 1                                                                            #### <SET> ####
weight_decay = 0.01
logging_steps = 10                                                                         #### <SET> ####
warmup_steps = 10                                                                           #### <SET> ####

#------------------------------- SPARSE ARGS -----------------------------------------------------------------

mask_scores_learning_rate = 1e-2

distil_alpha_ce = 0.1
distil_alpha_teacher = 0.9
distil_temperature = None

dense_pruning_method = "topK:1d_alt"                                                        #### <SET> ####
attention_pruning_method = "topK"                                                           #### <SET> ####
initial_threshold = 1.0                                                                     #### <SET> ####
final_threshold = 0.5                                                                      #### <SET> ####

attention_block_rows = 32                                                                   #### <SET> ####
attention_block_cols = 32                                                                   #### <SET> ####
dense_block_rows = None                                                                     #### <SET> ####
dense_block_cols = None                                                                     #### <SET> ####       

initial_warmup = 1
final_warmup = 3

regularization = "disabled"
regularization_final_lambda = None
attention_lambda = None
dense_lambda = None


#---------------------------------- PATH ----------------------------------------------------------------------

# Adding some context to output dump
# TEST modification
output_directory =   str("dump") + str("/") \
                   + str("trained-models") + str("/") \
                   + str(dataset_model_name_prefix) + str("/") \
                   + str("total_epochs_") + str(total_epochs) \
                   + str("_final_threshold_") + str(final_threshold) \
                   + str("_block_size_") + str(attention_block_rows) \
                   + str("_method_") + str(attention_pruning_method) \
                   + str("_") + str(timestr)                                                 

patched_model_save_location = os.path.join(output_directory, "patched_model")
save_model_fine_pruned_NOT_MASKED = os.path.join(output_directory, "fine-pruned-UNMASKED")
save_model_fine_pruned_MASKED = os.path.join(output_directory, "fine-pruned-MASKED")
cache_dir = os.path.join(output_directory, "model_cache_from_mpc")
logging_directory = os.path.join(output_directory, "logs")
output_metric_file = os.path.join(output_directory, "output_metrics.yaml")
normal_args_file = os.path.join(output_directory, "typical_training_args.yaml")
sparse_args_file = os.path.join(output_directory, "sparse_training_args.yaml")


#---------------------------------------------------------------------------------------------------------------









####################### MODEL TYPE SPECIFICS ####################################################################


#------------------------- DeiT Drop Token Variant I ---------------------------------------------------------------------

if model_to_utilize_for_experiment == "deit_drop_tokens_variant_i":
    info_dict_for_token_dropping = {                                                                   ##### <SET> if deit_drop_tokens_variant_i ######
                                        1:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True},
                                        
                                        4:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True},
                                        
                                        7:  {'keep_rate': 0.5,
                                            'fuse_inattentive': True}
                                    }
    token_drop_context_for_experiment = DropTokenLayerContextLayerwise(info_dict=info_dict_for_token_dropping)
else:
    token_drop_context_for_experiment = None