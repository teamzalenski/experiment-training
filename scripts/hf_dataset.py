# Lint as: python3
import datasets
import json
import logging
import os
import pandas as pd
from pathlib import Path
from datasets import ClassLabel, DownloadConfig
"""The Dataset."""

logger = datasets.logging.get_logger(__name__)

_CITATION = """"""

_DESCRIPTION = """"""

_URL = "https://huggingface.co/datasets/teamzalenski/astroentities/raw/main/"
_TRAINING_FILE = "train.jsonl"
_DEV_FILE = "validate.jsonl"
_TEST_FILE = "test.jsonl"


class Config(datasets.BuilderConfig):
    """The Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Config, self).__init__(**kwargs)

class HF(datasets.GeneratorBasedBuilder):
    """The Dataset."""

    BUILDER_CONFIGS = [
        Config(
            name="hf", version=datasets.Version("1.0.0"), description="The Dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                    "O",
                                    "B-Archive",
                                    "B-CelestialObject",
                                    "B-CelestialObjectRegion",
                                    "B-CelestialRegion",
                                    "B-Citation",
                                    "B-Collaboration",
                                    "B-ComputingFacility",
                                    "B-Database",
                                    "B-Dataset",
                                    "B-EntityOfFutureInterest",
                                    "B-Event",
                                    "B-Fellowship",
                                    "B-Formula",
                                    "B-Grant",
                                    "B-Identifier",
                                    "B-Instrument",
                                    "B-Location",
                                    "B-Mission",
                                    "B-Model",
                                    "B-ObservationalTechniques",
                                    "B-Observatory",
                                    "B-Organization",
                                    "B-Person",
                                    "B-Proposal",
                                    "B-Software",
                                    "B-Survey",
                                    "B-Tag",
                                    "B-Telescope",
                                    "B-TextGarbage",
                                    "B-URL",
                                    "B-Wavelength",
                                    "I-Archive",
                                    "I-CelestialObject",
                                    "I-CelestialObjectRegion",
                                    "I-CelestialRegion",
                                    "I-Citation",
                                    "I-Collaboration",
                                    "I-ComputingFacility",
                                    "I-Database",
                                    "I-Dataset",
                                    "I-EntityOfFutureInterest",
                                    "I-Event",
                                    "I-Fellowship",
                                    "I-Formula",
                                    "I-Grant",
                                    "I-Identifier",
                                    "I-Instrument",
                                    "I-Location",
                                    "I-Mission",
                                    "I-Model",
                                    "I-ObservationalTechniques",
                                    "I-Observatory",
                                    "I-Organization",
                                    "I-Person",
                                    "I-Proposal",
                                    "I-Software",
                                    "I-Survey",
                                    "I-Tag",
                                    "I-Telescope",
                                    "I-TextGarbage",
                                    "I-URL",
                                    "I-Wavelength"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "validate": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validate"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        print("Generating examples from = %s", filepath)
        labeled_paper = pd.read_json(path_or_buf=filepath, lines=True)
        current_tokens = []
        current_labels = []
        for i, p in labeled_paper.iterrows():
            for t in p['tokens']:
                current_tokens.append(str(t))
            for l in p['ner_tags']:
                current_labels.append(str(l))
            yield i, {
                "id": str(i),
                "tokens": current_tokens,
                "ner_tags": current_labels,
            }

class HFDataset(object):
    """
    """
    NAME = "HFDataset"

    def __init__(self):
        cache_dir = os.path.join(str(Path.home()), '.cache')
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        self._dataset = HF(cache_dir=cache_dir)
        print("Cache1 directory: ",  self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset["test"]

    def validate(self):
        return self._dataset["validate"]


if __name__ == '__main__':
    dataset = HFDataset().dataset
    print(len(dataset['train']))
    print(len(dataset['test']))
    print(len(dataset['validate']))
    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)
    print("First sample: ", dataset['train'][0])