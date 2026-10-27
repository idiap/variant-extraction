<!-- SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch> -->
<!-- SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch> -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# variant-extraction

Code for adapting small language models to extract genetic variants and associated genes from biomedical literature using a structured generation format.

## Task

Given a biomedical title and abstract, full-text passage, or annotated text segment, the model must output one line per extracted variant-gene pair:

```text
Variant: <mutation>, Gene: <gene>
```

If no variant is present, the required output is exactly:

```text
No variant
```

No explanations, JSON, Markdown, or additional free-form text should be produced by the model.

## Project overview

This repository includes code for:

- preprocessing heterogeneous biomedical variant-extraction datasets;
- supervised fine-tuning of small language models for structured variant extraction;
- domain-specific tokenizer extension with variant-aware tokens;
- ClinVar-based embedding warm-up for newly added tokens;
- contrastive post-training with plausible incorrect variant-gene outputs;
- evaluation of binary variant detection, variant extraction, and gene-variant pair extraction;
- diagnostic WikiText-2 perplexity evaluation to monitor general language-modeling degradation during fine-tuning.

## Models

The experiments use small language models from two open-weight model families:

- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

The two model families were selected to test whether adaptation effects are consistent across different backbones and tokenizers.

## Datasets

The extraction corpus is based on five biomedical variant-extraction resources:

- VariantScape
- tmVar v3
- Internal
- hereditary-disease entity-extraction corpus
- BRONCO

VariantScape, tmVar v3, Internal, and the hereditary-disease corpus are used to construct the merged in-distribution benchmark. BRONCO is held out as an out-of-distribution benchmark.

This repository provides preprocessing code and expected data formats but does not redistribute the external datasets listed in the License section below.

## Adaptation strategies

The repository supports the following configurations:

- zero-shot prompting;
- supervised fine-tuning baseline;
- VariantTokenizer with HGVS-related and amino-acid tokens;
- VariantTokenizer plus DNA-subsequence tokens;
- VariantTokenizer plus gene-symbol tokens;
- VariantTokenizer plus DNA and gene tokens;
- ClinVar embedding warm-up;
- contrastive post-training.

## Evaluation

The main extraction metrics are precision, recall, and F1 at three levels:

1. **Binary variant detection**: whether the model correctly identifies if an example contains at least one variant.
2. **Variant extraction**: whether the predicted variant strings match the gold variant mentions, ignoring gene assignment.
3. **Gene-variant pair extraction**: whether both the variant and its associated gene match as a pair.

Evaluation uses the same structured output schema across all model configurations.

## General language-modeling diagnostic

WikiText-2 perplexity is estimated before and during fine-tuning to assess whether specialization for structured extraction degrades broader language-modeling ability. This diagnostic is performed for the standard fine-tuned baseline and the simple VariantTokenizer configuration.

## Repository structure

```text
.
├── preprocessing/          # Dataset conversion and benchmark construction
├── training/               # Supervised fine-tuning and tokenizer-extension training
├── contrastive_learning/   # Contrastive dataset construction and post-training
├── evaluation/             # Extraction and perplexity evaluation scripts
├── resources/              # External resources and token lists, where available
├── scripts/                # Run scripts
└── README.md
```

The exact structure may differ depending on the released version of the repository.

## Installation

Create a Python environment and install the required packages:

```bash
pip install torch transformers datasets peft trl bitsandbytes pandas tqdm safetensors
```

Additional packages may be required depending on preprocessing scripts, hardware, and local configuration.

## Outputs

Extraction evaluation scripts generate:

```text
metrics.json
predictions.csv
confusion_matrix.csv
per_example_errors.csv
```

Perplexity evaluation scripts generate:

```text
metrics.json
summary.csv
predictions.csv
```

## License

The source code in this repository is licensed under the GNU General Public License v3.0 only (`GPL-3.0-only`).

This code makes use of the external models, datasets, and auxiliary resources listed below. *These resources are not redistributed as part of this repository.* Users must obtain them from their respective sources and comply with their applicable licenses, terms of use, and access conditions.

### Models

| Name | License | Link |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Qwen2.5-7B-Instruct | Apache License 2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |

### Datasets

#### Variant-extraction datasets

| Name | License | Link |
| --- | --- | --- |
| VariantScape | CC BY-NC 4.0 | https://zenodo.org/records/23466784 |
| Internal dataset | CC BY-NC 4.0 | https://zenodo.org/records/23470181 |
| tmVar v3 | Public domain (U.S. Government Work; no restrictions on use or reproduction imposed by NLM/U.S. Government) | https://github.com/ncbi/tmVar3 |
| Hereditary-disease entity-extraction corpus | No explicit dataset-level license identified; associated publication is CC BY 4.0 | https://github.com/dlhuang/Entity_Extraction_Hereditary_Disease_2023/tree/master/data |
| BRONCO | No explicit dataset-level license identified; associated publication is CC BY 4.0 and the corpus is stated to be freely available | http://infos.korea.ac.kr/bronco |

#### Auxiliary datasets and resources

| Name | License | Link |
| --- | --- | --- |
| ClinVar | Freely available; attribution to ClinVar requested when data are copied or redistributed | https://www.ncbi.nlm.nih.gov/clinvar/ |
| WikiText-2 | CC BY-SA 3.0 and GFDL | https://huggingface.co/datasets/Salesforce/wikitext |

The licenses and terms listed above apply to the corresponding external resources and do not imply that those resources are licensed under the GPL-3.0-only license of this repository.

## Contact

For questions, please contact the repository maintainers.
