from datasets import load_metric
import datasets
from transformers import AutoTokenizer

from hf_dataset import HFDataset

metric = load_metric("seqeval")
logger = datasets.logging.get_logger(__name__)

class HFTokenizer(object):
    NAME = "HFTokenizer"

    def __init__(self,
                 hf_pretrained_tokenizer_checkpoint):
        self.model_max_length=512
        self._tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_tokenizer_checkpoint,model_max_length=self.model_max_length)

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def init_vf(hf_pretrained_tokenizer_checkpoint):
        return HFTokenizer(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True):
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           padding="do_not_pad",
                                           truncation="only_first",
                                           is_split_into_words=True)
        labels = []
        label_ids = []
        ner_tags = examples["ner_tags"][0]
        for i in range(min(len(ner_tags),self.model_max_length)):
            label_ids.append(ner_tags[i])
        labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


if __name__ == '__main__':

    hf_pretrained_tokenizer_checkpoint = "distilbert-base-uncased"
    dataset = HFDataset().dataset

    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    tokenized_datasets = dataset.map(hf_preprocessor.tokenize_and_align_labels, batched=True)
    print(len(tokenized_datasets))