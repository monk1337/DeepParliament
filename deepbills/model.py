import numpy as np
import torch
import pandas as pd
from pathlib import Path
from deepbills.metrics import Score_Calculation
import torch.nn as nn
from datasets import ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import os


os.environ["WANDB_SILENT"] = "true"
class DeepBills(object):

    def __init__(self,
                dataset,
                model_name      = 'bert-base-uncased',
                portion_name    = 'first',
                fold_name       = 1,
                problem_name    = 'multiclass',
                batch_size      = 8, 
                epoch           = 10, 
                use_fp16        = True):

        self.dataset         = dataset
        self.model_name      = model_name
        self.problem_name    = problem_name
        self.epoch           = epoch
        self.batch_size      = batch_size
        self.use_fp16        = use_fp16
        self.portion_name    = portion_name
        self.fold_name       = fold_name
        
        
        if self.problem_name == 'multiclass':
          self.class_weights  = self.class_weights_cal()
        else:
          self.class_weights  = None
          
        
        folder_name_mix       = f'{self.problem_name}_{self.portion_name}_{str(self.fold_name)}'
        self.file_name        = f"models_output/{self.model_name}/{folder_name_mix}"
        Path(self.file_name).mkdir(parents=True, exist_ok=True)

        self.score_calc      = Score_Calculation(problem_type = self.problem_name, 
                                                 file_name    = self.file_name)
        
        
        unique_lab         = sorted(list(set(self.dataset['train']['labels'])))
        self.labels        = ClassLabel(names = unique_lab)
        id2label           = {key: label for key, label in enumerate(self.labels.names)}
        label2id           = {label: key for key, label in id2label.items()}
        self.logging_steps = len(self.dataset['train']) // self.batch_size

        
        self.tokenizer     = AutoTokenizer.from_pretrained(self.model_name)
        self.model         = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                           num_labels=len(self.labels.names),
                                                           id2label=id2label,
                                                           label2id=label2id)
        
        self.train_dataset = self.dataset.map(self.tokenize_text, batched=True)
        
        print("example of dataset", len(self.train_dataset['train'][10]['input_ids']))
        print("example of dataset", len(self.train_dataset['test'][10]['input_ids']))
        
        self.training_args = TrainingArguments(
                                                output_dir                   = self.file_name,
                                                num_train_epochs             = self.epoch,
                                                learning_rate                = 5e-05,
                                                per_device_train_batch_size  = self.batch_size,
                                                per_device_eval_batch_size   = self.batch_size,
                                                weight_decay                 = 0.01,
                                                evaluation_strategy          = 'epoch',
                                                logging_steps                = self.logging_steps,
                                                fp16                         = self.use_fp16,
                                                push_to_hub                  = False, 
                                                save_total_limit             = 1,
                                                save_strategy                = "no",
                                                load_best_model_at_end       = False,
                                                report_to                    = "wandb",
                                                )


        
        
        self.trainer       = WeightedTrainer(custom_class_weight = self.class_weights,
                                                model                = self.model,
                                                args                 = self.training_args, 
                                                compute_metrics      = self.score_calc.compute_metrics,
                                                train_dataset        = self.train_dataset['train'],
                                                eval_dataset         = self.train_dataset['test'],
                                                tokenizer            = self.tokenizer
                                                )
        

    def tokenize_text(self, batch):

        data_df            = self.tokenizer(batch['Text'], truncation=True, max_length=512)
        data_df['labels']  = self.labels.str2int(batch['labels'])
        return data_df


    def class_weights_cal(self):
        
        df_dataset         = self.dataset['train'].to_pandas()
        class_weights      = (1 - (df_dataset['labels'].value_counts().sort_index()/len(df_dataset))).values

        if self.use_fp16:
          class_weights = torch.from_numpy(class_weights).float().to("cuda")
        else:
          class_weights = torch.from_numpy(class_weights).float()
        
        return class_weights



class WeightedTrainer(Trainer):

    def __init__(self, custom_class_weight, **kwargs,):
        super().__init__(**kwargs)
        self.custom_class_weight = custom_class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs        = model(**inputs)
        logits         = outputs.get('logits')
        labels         = inputs.get('labels')
        loss_func      = nn.CrossEntropyLoss(weight = self.custom_class_weight)
        loss           = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss