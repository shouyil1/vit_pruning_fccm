from TEST_deit_blockprune_run_params import (
    teacher_constructor,
    output_directory, 
    distil_teacher_name_or_path, 
    patched_model_save_location, 
    save_model_fine_pruned_NOT_MASKED, 
    save_model_fine_pruned_MASKED, 
    logging_directory,
    num_gpus_used, 
    common_train_eval_batch_size_per_device, 
    learning_rate, 
    total_epochs, 
    weight_decay, 
    logging_steps, 
    warmup_steps,
    mask_scores_learning_rate,
    distil_alpha_ce, 
    distil_alpha_teacher, 
    distil_temperature, 
    dense_pruning_method, 
    attention_pruning_method, 
    initial_threshold, 
    final_threshold, 
    attention_block_rows,
    attention_block_cols, 
    dense_block_rows, 
    dense_block_cols, 
    initial_warmup, 
    final_warmup, 
    regularization, 
    regularization_final_lambda, 
    attention_lambda, 
    dense_lambda,
    output_metric_file,
    normal_args_file, 
    sparse_args_file,
    cache_dir,
    dataset_to_utilize_for_experiment,
    model_to_utilize_for_experiment,
    token_drop_context_for_experiment
)


from deit_token_drop_variant_i import DeiTForImageClassificationDropTokens as DeiTForImageClassificationDropTokens_Variant_I
from deit_token_drop_variant_i import DeiTConfigDropTokens as DeiTConfigDropTokens_Variant_I


import pandas as pd
import numpy as np

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

from datasets import load_dataset
from datasets import load_metric

from transformers import pipeline
from transformers import TrainingArguments, Trainer
from transformers import DeiTForImageClassification, DeiTImageProcessor
from transformers import ViTForImageClassification 
from transformers import AutoConfig

from deit_blockprune_trainer import DeiTSparseTrainingArguments, DeiTModelPatchingCoordinator, DeiTSparseTrainer

from PIL import Image

from sklearn.metrics import accuracy_score

import yaml


################### TODO ################################################################
'''

- ImageNet support
- Token drop addition and configuration in verbose dumps
- Dataset utilized in verbose dumps
- Model type in verbose dumps


'''
#########################################################################################

################### NOTE: Add dataset support ###########################################
if dataset_to_utilize_for_experiment != "cifar-10":
    raise Exception("Support for datasets other than cifar-10 not yet added!")
#########################################################################################

train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
split_to_train_val = train_ds.train_test_split(test_size=0.1)
train_ds = split_to_train_val['train']
val_ds = split_to_train_val['test']


id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}


processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")


image_mean, image_std = processor.image_mean, processor.image_std                 
size = processor.crop_size["height"]                                             



normalize = Normalize(mean=image_mean, std=image_std)                        
_train_transforms = Compose(
        [
            RandomResizedCrop(size),                                        
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),                                             
            ToTensor(),
            normalize,
        ]
    )


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples



train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms) 



def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))



#################### NOTE: Various MODEL support #######################################################
########################################################################################################
########################################################################################################

if model_to_utilize_for_experiment == "deit":
    model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224",
                                                        id2label=id2label,
                                                        label2id=label2id)
elif model_to_utilize_for_experiment == "deit_drop_tokens_variant_i":
    if token_drop_context_for_experiment is not None:
        
        baseline_config = AutoConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        
        dict_for_kwargs = baseline_config.__dict__
        
        del dict_for_kwargs['id2label']
        del dict_for_kwargs['label2id']
        
        config_drop_tokens = DeiTConfigDropTokens_Variant_I(token_dropping_info=token_drop_context_for_experiment,
                                                            kwargs=dict_for_kwargs,
                                                            id2label=id2label,
                                                            label2id=label2id)
        
        model = DeiTForImageClassificationDropTokens_Variant_I.from_pretrained(pretrained_model_name_or_path="facebook/deit-base-distilled-patch16-224",
                                                                               config=config_drop_tokens)
    else:
        raise Exception("Token drop context not found!")
else:
    raise Exception("Other models not supported at the moment!")


######################## Args ############################################################################################
metric_name = "accuracy"
save_strat = "epoch"
eval_strat = "epoch"
load_best_model_end = True
remove_unused_columns = False
per_device_batch_train = common_train_eval_batch_size_per_device        
per_device_batch_eval = common_train_eval_batch_size_per_device         
##########################################################################################################################

####################### Sparse args ######################################################################################
attention_output_with_dense = False
bias_mask = True
mask_init = "constant"
mask_scale = 0.0
eval_with_current_patch_params = False
ampere_pruning_method = "disabled"
qat = False
layer_norm_patch = False
gelu_patch = False
final_finetune = None
linear_min_parameters = None
qconfig = None
rewind_model_name_or_path = None
gelu_patch_steps = None
decoder_attention_lambda = None
decoder_dense_lambda = None
initial_ampere_temperature = None
final_ampere_temperature = None
layer_norm_patch_steps = None
layer_norm_patch_start_delta = None
##########################################################################################################################



args = TrainingArguments(
    output_dir = output_directory,
    save_strategy = save_strat,
    evaluation_strategy = eval_strat,
    learning_rate = learning_rate,
    per_device_train_batch_size = per_device_batch_train,
    per_device_eval_batch_size = per_device_batch_eval,
    num_train_epochs = total_epochs,
    weight_decay = weight_decay,
    load_best_model_at_end = load_best_model_end,
    metric_for_best_model = metric_name,
    logging_dir = logging_directory,
    remove_unused_columns = remove_unused_columns,
    logging_steps = logging_steps,                           
    warmup_steps = warmup_steps
)



param_dict = {
'mask_scores_learning_rate' : mask_scores_learning_rate,
'dense_pruning_method' : dense_pruning_method,
'attention_pruning_method' : attention_pruning_method,
'attention_output_with_dense' : attention_output_with_dense,
'bias_mask' : bias_mask,
'mask_init' : mask_init,
'mask_scale' : mask_scale,
'dense_block_rows' : dense_block_rows,
'dense_block_cols' : dense_block_cols,
'attention_block_rows' : attention_block_rows,
'attention_block_cols' : attention_block_cols,
'initial_threshold' : initial_threshold,
'final_threshold' : final_threshold,
'initial_warmup' : initial_warmup,
'final_warmup' : final_warmup,
'regularization' : regularization,
'regularization_final_lambda' : regularization_final_lambda,
'attention_lambda' : attention_lambda,
'dense_lambda' : dense_lambda,
'distil_teacher_name_or_path' : distil_teacher_name_or_path,
'distil_alpha_ce' : distil_alpha_ce,
'distil_alpha_teacher' : distil_alpha_teacher,
'distil_temperature' : distil_temperature,
'final_finetune' : final_finetune,
'linear_min_parameters' : linear_min_parameters,
'eval_with_current_patch_params' : eval_with_current_patch_params,

'ampere_pruning_method' : ampere_pruning_method,
'qat' : qat,
'qconfig' : qconfig,
'rewind_model_name_or_path' : rewind_model_name_or_path,
'layer_norm_patch' : layer_norm_patch,
'layer_norm_patch_steps' : layer_norm_patch_steps,
'layer_norm_patch_start_delta' : layer_norm_patch_start_delta,
'gelu_patch' : gelu_patch,
'gelu_patch_steps' : gelu_patch_steps,
'decoder_attention_lambda' : decoder_attention_lambda,
'decoder_dense_lambda' : decoder_dense_lambda,
'initial_ampere_temperature' : initial_ampere_temperature,
'final_ampere_temperature' : final_ampere_temperature,
}


sparse_args = DeiTSparseTrainingArguments()
for k,v in param_dict.items():
    if v is not None:
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k} despite value of argument being not None")






mpc = DeiTModelPatchingCoordinator(
    sparse_args = sparse_args, 
    device = torch.device("cuda"), 
    cache_dir = cache_dir, 
    logit_names = ["logits"],                                   
    teacher_constructor = teacher_constructor,
    model_name_or_path = "facebook/deit-base-distilled-patch16-224"
)


mpc.patch_model(model=model)
model.save_pretrained(patched_model_save_location)





class FinalPruningTrainer(DeiTSparseTrainer, Trainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        DeiTSparseTrainer.__init__(self, sparse_args)





trainer = FinalPruningTrainer(
                                sparse_args = sparse_args,
                                args = args,

                                model = model,
                                
                                train_dataset = train_ds,
                                eval_dataset = val_ds,

                                tokenizer = processor,
                                
                                data_collator = collate_fn,
                                compute_metrics = compute_metrics,
                            )


arg_set_normal = trainer.args.__dict__
arg_set_sparse = trainer.sparse_args.__dict__
with open(normal_args_file, 'w') as convert_file:
    convert_file.write(yaml.dump(arg_set_normal)) 
with open(sparse_args_file, 'w') as convert_file:
    convert_file.write(yaml.dump(arg_set_sparse)) 




trainer.set_patch_coordinator(mpc)

trainer.train()

trainer.save_model(save_model_fine_pruned_NOT_MASKED)




outputs = trainer.predict(test_ds)
output_metrics = outputs.metrics
with open(output_metric_file, 'w') as convert_file:
    convert_file.write(yaml.dump(output_metrics)) 


mpc.compile_model(trainer.model)

trainer.save_model(save_model_fine_pruned_MASKED)