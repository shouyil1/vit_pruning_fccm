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
from transformers import DeiTForImageClassificationWithTeacher
from transformers import ViTForImageClassification 
from transformers import AutoConfig

from PIL import Image

from sklearn.metrics import accuracy_score

import yaml

import os

import time


#############################################################################################
################################## Setting Parameters #######################################
#############################################################################################

timestr = time.strftime("%Y%m%d-%H%M%S")
output_directory = str("trained-models/deit-small-baseline-imagenet-1k") + str("-") + str(timestr)
logging_directory = os.path.join(output_directory, "logs")
normal_args_file = os.path.join(output_directory, "typical_training_args.yaml")
save_model_best_baseline_loc = os.path.join(output_directory, "best-baseline-loaded-at-end")
output_metric_file = os.path.join(output_directory, "output_metrics.yaml")

num_gpus_used = 4 
common_train_eval_batch_size_per_device = 32
learning_rate = 2e-5
total_epochs = 10
weight_decay = 0.01
logging_steps = 1000

#############################################################################################


# Baseline runner for DeiT Small model


model_to_utilize_for_experiment = "deit"
deit_model_size_to_use_for_experiment = "small"
path_to_model_to_use_for_experiment = "facebook/deit-small-distilled-patch16-224"
dataset_to_utilize_for_experiment = "imagenet-1k"


##############################################################################################








train_ds, test_ds = load_dataset("imagenet-1k", split=['train', 'validation'])   # Validation is TEST

splitter_to_train_val = train_ds.train_test_split(test_size=25000)    # 25k images from train as part of val            

train_ds = splitter_to_train_val['train']       # 1256167    instances
val_ds = splitter_to_train_val['test']          # 25000      instances

key_to_get_image = 'image'

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id, label in id2label.items()}












processor = DeiTImageProcessor.from_pretrained(path_to_model_to_use_for_experiment)          # note that image processor for DeiT and DeiT drop tokens don't have any difference; drop tokens context is not needed for the processor

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
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples[key_to_get_image]]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples[key_to_get_image]]
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










model = DeiTForImageClassification.from_pretrained(path_to_model_to_use_for_experiment,
                                                   id2label=id2label,
                                                   label2id=label2id)






################################ NOTE ##################################################################
############ We do not attach forcibly the classifier weights to gain some unfair advantage ############
########################################################################################################



metric_name = "accuracy"
save_strat = "epoch"
eval_strat = "epoch"
load_best_model_end = True
remove_unused_columns = False
per_device_batch_train = common_train_eval_batch_size_per_device        
per_device_batch_eval = common_train_eval_batch_size_per_device 



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
    #warmup_steps = warmup_steps                              ######### NO WARM-UP; NET REDUCED EPOCHS #########
)




trainer = Trainer          (    
                                
                                args = args,

                                model = model,
                                
                                train_dataset = train_ds,
                                eval_dataset = val_ds,

                                tokenizer = processor,
                                
                                data_collator = collate_fn,
                                compute_metrics = compute_metrics,

                            )





arg_set_normal = trainer.args.__dict__
arg_set_normal['MODEL_VARIANT_CLASS_NAME_FOR_EXPERIMENT'] = model_to_utilize_for_experiment
arg_set_normal['DATASET_NAME_FOR_EXPERIMENT'] = dataset_to_utilize_for_experiment
arg_set_normal['USED_MODEL_PATH_FOR_FROM_PRETRAINED'] = path_to_model_to_use_for_experiment




with open(normal_args_file, 'w') as convert_file:
    convert_file.write(yaml.dump(arg_set_normal)) 




trainer.train()



trainer.save_model(save_model_best_baseline_loc)



outputs = trainer.predict(test_ds)
output_metrics = outputs.metrics
with open(output_metric_file, 'w') as convert_file:
    convert_file.write(yaml.dump(output_metrics)) 







################# END #####################################################