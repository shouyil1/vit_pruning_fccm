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
from transformers import DeiTForImageClassification, DeiTImageProcessor, DeiTModel
from transformers import ViTForImageClassification 

from deit_blockprune_trainer import DeiTSparseTrainingArguments, DeiTModelPatchingCoordinator, DeiTSparseTrainer
from inference_patchers_blockprune import optimize_model_deit

from PIL import Image

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from time import perf_counter

from contextlib import redirect_stdout

import os

import yaml
import json
from pathlib import Path


###################################################################
########################### TODO ##################################
###################################################################
'''

Dynamic TODO; current status: outdated file

NOTE: This file is outdated; revision required


'''


model_names_path_list = [
                "deit-cifar10-blockprune-topK-block-8-epoch-10-finalthreshold-0.2-20231204-035948",
                "deit-cifar10-blockprune-topK-block-8-epoch-10-finalthreshold-0.3-20231204-021427",
                "deit-cifar10-blockprune-topK-block-8-epoch-10-finalthreshold-0.4-20231203-232533",
                "deit-cifar10-blockprune-topK-block-8-epoch-10-finalthreshold-0.5-20231203-213500",
               
                "deit-cifar10-blockprune-topK-block-16-epoch-10-finalthreshold-0.2-20231203-184807",
                "deit-cifar10-blockprune-topK-block-16-epoch-10-finalthreshold-0.3-20231201-060301",
                "deit-cifar10-blockprune-topK-block-16-epoch-10-finalthreshold-0.4-20231201-040427",
                "deit-cifar10-blockprune-topK-block-16-epoch-10-finalthreshold-0.5-20231201-005935",

                "deit-cifar10-blockprune-topK-block-32-epoch-10-finalthreshold-0.2-20231129-224744",
                "deit-cifar10-blockprune-topK-block-32-epoch-10-finalthreshold-0.3-20231129-201650",
                "deit-cifar10-blockprune-topK-block-32-epoch-10-finalthreshold-0.4-20231129-175805",
                "deit-cifar10-blockprune-topK-block-32-epoch-10-finalthreshold-0.5-20231129-175552",
                
                "deit-cifar10-blockprune-topK-block-64-epoch-10-finalthreshold-0.2-20231130-225820",
                "deit-cifar10-blockprune-topK-block-64-epoch-10-finalthreshold-0.3-20231130-200526",
                "deit-cifar10-blockprune-topK-block-64-epoch-10-finalthreshold-0.4-20231130-174443",
                "deit-cifar10-blockprune-topK-block-64-epoch-10-finalthreshold-0.5-20231130-155452"
                ]



map_to_model_type = [{"block": x, "threshold": y} for x,y in zip([8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64], [0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5])]

prefix = "fine-pruned-deit-cifar10-models"

postfix = "fine-pruned-MASKED"

postfix_to_output_metrics = "output_metrics.yaml"


some_layer_names_for_viz =  [
                                "deit.encoder.layer.11.attention.attention.query.weight",       # Q
                                "deit.encoder.layer.11.attention.attention.key.weight",         # K
                                "deit.encoder.layer.11.attention.attention.value.weight",       # V
                                "deit.encoder.layer.11.attention.output.dense.weight",          # O

                                "deit.encoder.layer.11.intermediate.dense.weight",              # FFN 
                                "deit.encoder.layer.11.output.dense.weight"                     # FFN
                            ]


target_local_directory = "results_block_pruning_dense_remove"










# Generic Processor and Tester (Specific not needed most likely should be identical)
_, test_ds = load_dataset('cifar10', split=['train', 'test'])
processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")


image_mean, image_std = processor.image_mean, processor.image_std                 # Mean and STD
size = processor.crop_size["height"]                                              # 224 X 224 for the ViT/DeiT
normalize = Normalize(mean=image_mean, std=image_std)

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),                                                # CLAIM: Is redundant
            ToTensor(),
            normalize,
        ]
    )

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


test_ds.set_transform(val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


id2label = {id:label for id, label in enumerate(test_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}


def compute_latencies(model, device, batch_size):
    inputs = collate_fn([test_ds[i] for i in range(batch_size)])
    latencies = []
    
    interim_dict = {}

    for k, v in inputs.items():
        interim_dict[k] = v.to(device)

    inputs = interim_dict

    model = model.to(device)

    # Warmup
    for _ in range(5):
        _ = model(**inputs)
        
    for _ in range(10):
        start_time = perf_counter()
        _ = model(**inputs)
        latency = perf_counter() - start_time 
        latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies) 
    print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}") 
    return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

cpu_latency_batch_size = 25
gpu_latency_batch_size = 25

allow_verbose_dump = False

result_dict = {}


# Baseline
baseline_path = "deit-baseline-cifar10/checkpoint-7500"
baseline_model = DeiTForImageClassification.from_pretrained(baseline_path,
                                                            id2label=id2label,
                                                            label2id=label2id)
baseline_accuracy = 0.9856          # Ran before

baseline_parameters = baseline_model.num_parameters()



for idx_to_type, current_model_path_midfix in enumerate(model_names_path_list):

    full_current_model_path = os.path.join(*[prefix, current_model_path_midfix, postfix])

    current_model = DeiTForImageClassification.from_pretrained(full_current_model_path)

    block, threshold = map_to_model_type[idx_to_type]['block'], map_to_model_type[idx_to_type]['threshold']

    dumping_verbose_file_path = os.path.join(*[target_local_directory, "some_verbose_dumps", str(str("block_") + str(block) + str("_threshold_") + str(threshold) + str(".txt"))])

    if allow_verbose_dump == True:     # since this stays same over parameterized GPU/CPU batch size for inference runs, we allow to not re-dump
        with open(dumping_verbose_file_path, "w") as f:
            with redirect_stdout(f):
                redacted_model = optimize_model_deit(model=current_model,
                                                mode="dense")
    else:
        redacted_model = optimize_model_deit(model=current_model,
                                                mode="dense")
    
    latencies_GPU = compute_latencies(model = redacted_model, 
                                      device = torch.device('cuda:0'),
                                      batch_size = gpu_latency_batch_size)
    
    latencies_CPU = compute_latencies(model = redacted_model,
                                      device = torch.device('cpu'),
                                      batch_size = cpu_latency_batch_size)

    effective_density_model = redacted_model.num_parameters()/current_model.num_parameters()     # Unfair comparison since removed heads won't be "counted"

    effective_density_baseline = redacted_model.num_parameters()/baseline_parameters

    path_to_metrics = os.path.join(*[prefix, current_model_path_midfix, postfix_to_output_metrics])
    
    output_metrics_info = yaml.safe_load(Path(path_to_metrics).read_text())

    model_accuracy = output_metrics_info["test_accuracy"]

    result_dict[idx_to_type] = {
        "block_size": block,
        "threshold": threshold,
        "model_accuracy": model_accuracy,
        "GPU_latency_ms:": latencies_GPU['time_avg_ms'],
        "CPU_latency_ms:": latencies_CPU['time_avg_ms'],
        "effective_density_RELATIVE": effective_density_model,
        "effective_density_BASELINE": effective_density_baseline,
        "batch_size_GPU": gpu_latency_batch_size,
        "batch_size_CPU": cpu_latency_batch_size 
    }

baseline_latency_GPU = compute_latencies(model=baseline_model,
                                         device=torch.device('cuda:0'),
                                         batch_size = gpu_latency_batch_size)

baseline_latency_CPU = compute_latencies(model=baseline_model, 
                                         device=torch.device('cpu'),
                                         batch_size=cpu_latency_batch_size)

result_dict["baseline"] = {
    "block_size": "NA",
    "threshold": "NA",
    "model_accuracy": baseline_accuracy,
    "GPU_latency_ms:": baseline_latency_GPU['time_avg_ms'],
    "CPU_latency_ms:": baseline_latency_CPU['time_avg_ms'],
    "effective_density_RELATIVE": 1,
    "effective_density_BASELINE": 1,
    "batch_size_GPU": gpu_latency_batch_size,
    "batch_size_CPU": cpu_latency_batch_size 
}

# Parameterizing saves on GPU & CPU inference batch size
compiled_result_file_path = os.path.join(*[target_local_directory, str(str("GPU_") + str(gpu_latency_batch_size) + str("_CPU_") + str(cpu_latency_batch_size) + str(".json"))])

with open(compiled_result_file_path, 'w') as convert_file:
    convert_file.write(json.dumps(result_dict))      