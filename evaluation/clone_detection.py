# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import os
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy
from datasets import ClassLabel, load_dataset
from transformers import (
    logging,
    set_seed,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="codesage_small")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_gpu_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze", type=str2bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--model_hub_name", type=str, default="codeclone_model")
    return parser.parse_args()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": np.mean(predictions == labels)}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


def main():
    args = get_args()
    set_seed(args.seed)

    ds = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench", cache_dir="./tmp/")
    labels = ClassLabel(num_classes=2, names=[True, False])
    ds = ds.cast_column("label", labels)

    print("Loading config, model, and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.problem_type = "single_label_classification"
    config.num_labels = 2
    config.classifier_dropout = None
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.freeze:
        print("Freezing model parameters")
        for param in model.roberta.parameters():
            param.requires_grad = False

    def convert_examples_to_features(example):
        inputs = tokenizer(
            example["func1"], example["func2"], truncation=True, max_length=args.max_seq_length,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    tokenized_datasets = ds.map(
        convert_examples_to_features,
        batched=True,
        remove_columns=["id", "id1", "id2", "func1", "func2"],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.per_gpu_batch_size,
        per_device_eval_batch_size=args.per_gpu_batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        metric_for_best_model="accuracy",
        run_name="code-clone-java"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Evaluation accuracy on the test set: {result['eval_accuracy']}")

    # push the model to the Hugging Face hub
    if args.push_to_hub:
        model.push_to_hub(args.model_hub_name)


if __name__ == "__main__":
    main()
