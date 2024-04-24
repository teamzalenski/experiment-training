import datasets
from datasets import load_dataset, load_metric, ClassLabel, DownloadConfig
import numpy as np
import os
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from pathlib import Path
from hf_tokenize import HFTokenizer
from hf_dataset import HFDataset

metric = load_metric("seqeval")

logger = datasets.logging.get_logger(__name__)

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    model_n_version = "hf"
    max_epochs = 150
    learning_rate = 2e-5
    batch_size = 16
    model_root_dir = "."
    hf_pretrained_model_checkpoint = "distilbert-base-uncased"
    hf_pretrained_tokenizer_checkpoint = "distilbert-base-uncased"
    hf_dataset = HFDataset()
    dataset = HFDataset().dataset
    hf_model = AutoModelForTokenClassification.from_pretrained(hf_pretrained_model_checkpoint,
                                                    num_labels=len(hf_dataset.labels))
    hf_model.config.id2label = hf_dataset.id2label
    hf_model.config.label2id = hf_dataset.label2id
    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)
    train_tokenized_datasets = hf_dataset.dataset['train'].map(hf_preprocessor.tokenize_and_align_labels, batched=False)
    validate_tokenized_datasets = hf_dataset.dataset['validation'].map(hf_preprocessor.tokenize_and_align_labels, batched=False)
    args = TrainingArguments(
        f"hf",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
    )
    data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)
    trainer = Trainer(
        hf_model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=validate_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=hf_preprocessor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p=p, label_list=hf_dataset.labels)
    )
    trainer.train()
    trainer.evaluate()
    test_tokenized_datasets = hf_dataset.dataset['test'].map(hf_preprocessor.tokenize_and_align_labels, batched=False)
    predictions, labels, _ = trainer.predict(test_tokenized_datasets)
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [hf_dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [hf_dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
    out_dir = os.path.expanduser(model_root_dir) + "/" + model_n_version
    trainer.save_model(out_dir)
