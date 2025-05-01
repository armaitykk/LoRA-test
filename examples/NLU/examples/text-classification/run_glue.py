#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

#!/usr/bin/env python
# coding=utf-8

# --- Imports ---
import logging
import os
import random
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from evaluate import load as load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
check_min_version("4.4.0")

# --- Dynamic LoRA r Search ---
if model_args.lora_r == -1:
    auto_r_values = [2, 4, 8, 16, 32]  # Dynamic r values to test
    best_r = None
    best_score = -float("inf")
    results = {}

    # Preprocess datasets (you must do this before the loop)
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Define metric computation
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Loop over LoRA r values
    for r_val in auto_r_values:
        print(f"\n=== Trying LoRA rank r={r_val} ===")

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            cls_dropout=0.1,
            apply_lora=model_args.apply_lora,
            lora_alpha=model_args.lora_alpha,
            lora_r=r_val,
            apply_adapter=model_args.apply_adapter,
            adapter_type=model_args.adapter_type,
            adapter_size=model_args.adapter_size,
            reg_loss_wgt=model_args.reg_loss_wgt,
            masking_prob=model_args.masking_prob,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        eval_score = eval_metrics.get("eval_accuracy") or eval_metrics.get("eval_matthews_correlation")

        print(f"Validation Score for r={r_val}: {eval_score}")
        results[r_val] = eval_score

        if eval_score > best_score:
            best_score = eval_score
            best_r = r_val

    print(f"\nBest LoRA rank: r={best_r} with score={best_score}")

    # === Plot LoRA Rank vs Score ===
    r_vals = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, scores, marker='o', linestyle='-', color='blue')
    plt.title('Validation Accuracy vs. LoRA Rank (r)')
    plt.xlabel('LoRA Rank (r)')
    plt.ylabel('Validation Score')
    plt.grid(True)
    plt.xticks(r_vals)
    plt.savefig("lora_rank_vs_accuracy.png")
    plt.show()

    # === Custom Inference Timing ===
    print("\n=== Running Custom Inference Timing ===")

    sample_sentence = "The ship sank beneath the waves."
    inputs = tokenizer(sample_sentence, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start_pre = time.time()
    _ = tokenizer(sample_sentence, return_tensors="pt", padding=True, truncation=True)
    end_pre = time.time()

    start_inf = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_inf = time.time()

    start_post = time.time()
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    end_post = time.time()

    print(f"\nPrediction: {prediction} (1 = acceptable, 0 = not acceptable)")
    print(f"Preprocessing time: {end_pre - start_pre:.4f} sec")
    print(f"Inference time:     {end_inf - start_inf:.4f} sec")
    print(f"Postprocessing time:{end_post - start_post:.4f} sec\n")

    times = [end_pre - start_pre, end_inf - start_inf, end_post - start_post]
    labels = ['Preprocessing', 'Inference', 'Postprocessing']

    plt.bar(labels, times)
    plt.title('Inference Timing Breakdown')
    plt.ylabel('Seconds')
    plt.savefig("inference_timing_chart.png")
    plt.show()

    return  # end early after r search to avoid re-training

