import json
import random
from typing import Optional
import csv
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertTokenizerFast
import numpy as np
import ast
import pandas as pd
import pickle

class ClassifcationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        task: str,
        max_seq_len: int = 512,
        all_examples_with_null: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.all_examples_with_null = all_examples_with_null
        self.task = task

    def __call__(self, data):
        admission_notes = [x["text"] for x in data]

        labels = torch.stack([x["labels"] for x in data])

        sample_ids = (
            [str(x["sample_id"]) for x in data] if self.task in ["dia", "pro"] else None
        )
        tokenized = self.tokenizer(
            admission_notes,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [
            torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]
        ]
        lengths = torch.tensor([len(x) for x in input_ids])
        # TODO: Hotfix hack for DisitlBertTokenizer 🤢 
        token_type_ids = (
            [torch.tensor(x) for x in tokenized["token_type_ids"]]
            if self.tokenizer.__class__ != DistilBertTokenizerFast
            else []
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        if token_type_ids != []:
            token_type_ids = torch.nn.utils.rnn.pad_sequence(
                token_type_ids, batch_first=True
            )

        tokens = [x.tokens for x in tokenized.encodings]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "token_type_ids": token_type_ids,
            "lengths": lengths,
            "tokens": tokens,
            "sample_ids": sample_ids,
        }


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, task, label_lookup, sampling_strategy: str = "random"):
        # tokenize admission notes
        self.task = task
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}
        self.sampling_strategy = sampling_strategy
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples.loc[idx]
        note = example["text"]
        labels = example["labels"]
        if self.task in ["dia", "pro"]:
            sample_id = example["hadm_id"]
            # hadm_id = example['hadm_id']
            label_ids = [self.label_lookup[x] for x in labels]
            label_idxs = torch.tensor([int(x) for x in label_ids])
            labels = torch.zeros(len(self.label_lookup), dtype=torch.float32)
            labels[label_idxs] = 1
            return {"text": note, "labels": labels, "sample_id": sample_id}
        elif self.task in ["los", "mp"]:
            label_arr = (
                torch.zeros(4, dtype=torch.float32)
                if self.task == "los"
                else torch.zeros(2, dtype=torch.float32)
            )
            label_arr[labels] = 1
            return {
                "text": note,
                "labels": label_arr,
            }
        elif self.task in ['pr']:
            label_arr = torch.zeros(len(self.label_lookup), dtype=torch.float32)
            label_arr[self.label_lookup[labels]] = 1
            return {
                "text": note,
                "labels": label_arr,
            }


class MIMICClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        use_code_descriptions: bool = False,
        data_dir: str = "/pvc/data/mimic-iv-processed/icd10/hosp/",
        task: str = "dia",
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 8,
        sampling_strategy: str = "random",
        val_sampling_strategy: str = "random",
        max_seq_len: int = 512,
        test_file=None,
        train_file=None,
        val_file=None,
        all_labels_path=None,
    ):
        super().__init__()

        if task in ["dia", "pro"]:
            test_data = pd.read_csv(data_dir + task + "/test_fold_1_simplified_true.csv", index_col="index")
            #test_data = pd.read_csv(test_file, index_col="index")
            test_data.labels = test_data.labels.map(lambda x: ast.literal_eval(x))
            validation_data = pd.read_csv(data_dir + task + "/dev_fold_1_simplified_true.csv", index_col="index")
            #validation_data = pd.read_csv(val_file, index_col="index")
            validation_data.labels = validation_data.labels.map(lambda x: ast.literal_eval(x))
            training_data = pd.read_csv(data_dir + task + "/train_fold_1_simplified_true.csv", index_col="index")
            #training_data = pd.read_csv(train_file, index_col="index")
            training_data.labels = training_data.labels.map(
                lambda x: ast.literal_eval(x)
            )
            if not all_labels_path:
                all_labels_path = data_dir + task + "/sorted_labels_simplified.txt"
                with open(data_dir + task + "/sorted_labels_simplified.txt") as f:
                    all_labels = f.read()
                all_labels = all_labels.split(",")
            else:
                with open(all_labels_path, "rb") as f:
                    all_labels = pickle.load(f)
            label_idx = {v: k for k, v in enumerate(all_labels)}
            
            training_data["hadm_id"] = training_data.index
            self.training_data = training_data[
                ["text", "labels", "hadm_id"]
            ].reset_index(drop=True)
            test_data["hadm_id"] = test_data.index
            self.test_data = test_data[["text", "labels", "hadm_id"]].reset_index(
                drop=True
            )
            validation_data["hadm_id"] = validation_data.index
            self.val_data = validation_data[["text", "labels", "hadm_id"]].reset_index(
                drop=True
            )
            self.use_code_descriptions = use_code_descriptions

        elif task in ["los", "mp"]:
            test_data = pd.read_csv(data_dir + task + "/test.csv").rename(
                columns={"class": "labels"}
            )
            train_data = pd.read_csv(data_dir + task + "/train.csv").rename(
                columns={"class": "labels"}
            )
            val_data = pd.read_csv(data_dir + task + "/val.csv").rename(
                columns={"class": "labels"}
            )
            self.training_data = train_data
            self.test_data = test_data
            self.val_data = val_data
            all_labels = list(train_data.labels.unique())
            all_labels.sort()
            label_idx = {v: k for k, v in enumerate(all_labels)}
        else:
            test_data = pd.read_csv(data_dir + task + "/test.csv").rename(
                columns={"careunit": "labels"}
            )
            train_data = pd.read_csv(data_dir + task + "/train.csv").rename(
                columns={"careunit": "labels"}
            )
            val_data = pd.read_csv(data_dir + task + "/val.csv").rename(
                columns={"careunit": "labels"}
            )
            self.training_data = train_data
            self.test_data = test_data
            self.val_data = val_data
            all_labels = list(train_data.labels.unique())
            all_labels.sort()
            label_idx = {v: k for k, v in enumerate(all_labels)}

        # build label index
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = ClassifcationCollator(self.tokenizer, task, max_seq_len)
        self.num_workers = num_workers
        self.task = task
        self.sampling_strategy = sampling_strategy
        self.val_sampling_strategy = val_sampling_strategy

    def setup(self, stage: Optional[str] = None):
        mimic_train = ClassificationDataset(
            self.training_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.sampling_strategy,
            task=self.task,
        )
        
        mimic_val = ClassificationDataset(
            self.val_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
            task=self.task,
        )

        mimic_test = ClassificationDataset(
            self.test_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
            task=self.task,
        )
        self.mimic_train = mimic_train
        self.mimic_val = mimic_val
        self.mimic_test = mimic_test
        print("Val length: ", len(self.mimic_val))
        print("Train Length: ", len(self.mimic_train))

    def train_dataloader(self):
        return DataLoader(
            self.mimic_train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mimic_val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mimic_test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
