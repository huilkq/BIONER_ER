import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{krallinger2015chemdner,
  title={The CHEMDNER corpus of chemicals and drugs and its annotation principles},
  author={Krallinger, Martin and Rabal, Obdulia and Leitner, Florian and Vazquez, Miguel and Salgado, David and Lu, Zhiyong and Leaman, Robert and Lu, Yanan and Ji, Donghong and Lowe, Daniel M and others},
  journal={Journal of cheminformatics},
  volume={7},
  number={1},
  pages={1--17},
  year={2015},
  publisher={BioMed Central}
}
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = ""
_URL = "https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/BC5CDR-IOB/"
_TRAINING_FILE = "train.tsv"
_DEV_FILE = "devel.tsv"
_TEST_FILE = "test.tsv"


class BC4CHEMDConfig(datasets.BuilderConfig):
    """BuilderConfig for  BC4CHEMD"""

    def __init__(self, **kwargs):
        """BuilderConfig for  BC4CHEMD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BC4CHEMDConfig, self).__init__(**kwargs)


class  BC4CHEMD(datasets.GeneratorBasedBuilder):
    """ BC4CHEMD dataset."""

    BUILDER_CONFIGS = [
        BC4CHEMDConfig(name="BC5CDR-Disease", version=datasets.Version("1.0.0"), description=" BC5CDR-Disease dataset"),
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
                                "B-Disease",
                                "I-Disease",
                                "B-Chemical",
                                "I-Chemical"
                                 
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        print(tokens)
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }