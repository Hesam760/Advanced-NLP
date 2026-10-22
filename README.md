# Advanced NLP — Multi-Hop Question Answering with Dense Retrieval

**Phase 1: Evaluation Infrastructure and Dense Retrieval**

This repository contains the implementation and evaluation pipeline for a multi-hop question-answering task developed as part of an Advanced Natural Language Processing course project.

The project establishes reproducible evaluation baselines, implements dense passage retrieval, compares retrieval-based and retrieval-free question-answering systems, and evaluates answer quality and retrieval performance on development data.

## Table of Contents

- [Project Overview](#project-overview)
- [Implemented Systems](#implemented-systems)
- [Models and Configuration](#models-and-configuration)
- [Retrieval and Evaluation](#retrieval-and-evaluation)
- [Repository Structure](#repository-structure)
- [Requirements and Setup](#requirements-and-setup)
- [Running the Project](#running-the-project)
- [Prediction Format](#prediction-format)
- [Results](#results)
- [Reproducibility](#reproducibility)

## Project Overview

Multi-hop question answering requires a system to combine information from multiple documents to answer questions that cannot always be resolved from a single passage.

This project investigates three complementary approaches:

1. **Closed-book question answering:** Answering questions using the language model's internal knowledge without retrieved passages.
2. **Oracle-context question answering:** Generating answers using the gold supporting documents supplied with each development question.
3. **Dense Retrieval-Augmented Generation (Dense RAG):** Retrieving relevant passages using dense embeddings and generating answers from the retrieved context.

The main objectives are to establish comparable baselines, evaluate retrieval quality independently from answer generation, identify common failure modes, and provide a consistent framework for comparing the systems.

## Implemented Systems

### 1. Closed-Book QA

The language model answers each question without access to retrieved passages.

This baseline provides a reference for evaluating how well the model answers questions using its parametric knowledge alone.

### 2. Oracle-Context QA

The language model receives the gold supporting documents associated with each development question.

This system helps investigate answer generation when the intended supporting evidence is provided directly. It serves as a diagnostic reference rather than an independent retrieval system.

Oracle-context evaluation is performed on the development split.

### 3. Dense RAG

Dense RAG retrieves relevant passages from the document collection and supplies them to the language model as context.

The retrieval pipeline uses dense vector representations and exact cosine-similarity search. The implementation also supports document masking before top-\(k\) selection, allowing retrieval to be restricted to eligible documents when required.

This system is evaluated on both retrieval quality and final answer quality.

## Models and Configuration

The final experiment configuration uses the following models:

| Component | Model |
|---|---|
| Language model | `Qwen/Qwen3.5-9B` |
| Dense retriever embedding model | `Qwen/Qwen3-Embedding-4B` |

The language model is configured for four-bit NF4 quantization with FP16 computation in the reported experimental setup.

The main system configurations are maintained separately:

- `configs/closed_book.yaml`
- `configs/oracle.yaml`
- `configs/dense.yaml`

These configuration files centralize system-specific settings, including model parameters, retrieval options, and generation settings.

The final run artifacts are intended to include a `run_config.yaml` alongside each `predictions.jsonl` file to record the configuration associated with that run.

## Retrieval and Evaluation

### Dense Retrieval

The retrieval pipeline uses normalized embeddings and cosine similarity to rank candidate passages.

The implementation includes:

- Exact cosine-similarity search.
- Top-\(k\) passage retrieval.
- Document masking before candidate selection.
- Evaluation against gold supporting-document identifiers.

### Answer Evaluation

Answer quality is measured using:

- **Exact Match (EM):** Whether the normalized prediction matches the reference answer.
- **Token-level F1:** The overlap between the predicted and reference answers.

### Retrieval Evaluation

The project implements retrieval metrics independently and compares them against the reference evaluation implementation.

The evaluated metrics include:

- Recall@\(k\)
- HitAll@\(k\)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)

The evaluation also examines performance by question type, including comparison, bridge comparison, compositional, and inference questions.

Paired comparisons and confidence intervals are used to analyze differences between systems.

## Repository Structure

The main project components are organized as follows:

```text
Advanced-NLP/
├── advanced-nlp.ipynb
├── configs/
│   ├── closed_book.yaml
│   ├── oracle.yaml
│   └── dense.yaml
├── src/
│   ├── config.py
│   ├── metrics.py
│   └── retrieval.py
├── tests/
├── runs/
│   ├── dev/
│   │   ├── closed_book/
│   │   ├── oracle/
│   │   └── dense/
│   └── hidden/
│       ├── closed_book/
│       └── dense/
└── README.md
```

The repository may also contain evaluation scripts and additional supporting files.

## Requirements and Setup

The experiments use Python, PyTorch, Hugging Face Transformers, and the supporting libraries specified in the notebook setup cells.

A compatible CUDA-enabled environment is recommended for running the language model efficiently. The reported experiments use a Kaggle environment with NVIDIA Tesla T4 GPUs.

### Getting the Repository

```bash
git clone https://github.com/Hesam760/Advanced-NLP.git
cd Advanced-NLP
```

### Environment Setup

Use the package installation commands provided in the setup cells of `advanced-nlp.ipynb` and ensure that the required course data and configuration files are available.

The notebook may contain environment-specific paths. Update those paths when running outside the original Kaggle environment.

## Running the Project

The main experimental workflow is provided in `advanced-nlp.ipynb`.

The general workflow is:

1. Install the required dependencies.
2. Load the course dataset and document collection.
3. Load the system configurations.
4. Initialize the language model and, where required, the embedding model.
5. Run the retrieval and answer-generation pipelines.
6. Save predictions in the required JSONL format.
7. Validate prediction files and evaluate the development results.
8. Analyze system differences and representative failure cases.

The notebook should be executed in order, with the required data files and model access available.

## Prediction Format

Each prediction is stored as one JSON object per line in a `predictions.jsonl` file.

A typical record contains:

```json
{
  "qid": "dev_001",
  "answer": "Example answer",
  "retrieved": ["doc_12", "doc_35"]
}
```

The example above illustrates the schema; the actual question identifiers and document identifiers depend on the dataset.

For the closed-book system, the `retrieved` field is empty because no passages are retrieved. For retrieval-based systems, it records the retrieved document identifiers.

Development and hidden predictions are stored separately. Hidden predictions should not be treated as independently evaluated results unless the corresponding reference labels are available.

## Results

The development evaluation covers 300 questions.

The reported experiments compare the three systems using answer-quality metrics and, for Dense RAG, retrieval metrics. Results are also analyzed by question type to investigate the strengths and limitations of dense retrieval and answer generation.

The detailed numerical results, confidence intervals, paired comparisons, and error analyses are documented in the project report.

## Reproducibility

Reproducibility is supported through:

- Centralized YAML configuration files.
- Separate prediction artifacts for each system and split.
- Run-specific configuration records.
- Independent retrieval-metric implementations and reference validation.
- Automated tests for core project functionality.
- A documented experimental workflow in the main notebook.

Model versions, generation settings, retrieval parameters, runtime environment, and dataset versions should be recorded when producing new experimental results.

## Scope

This repository focuses on the first phase of the project: evaluation infrastructure, baseline question answering, dense retrieval, and comparative analysis.

Additional experiment tracking, configuration-management extensions, and other optional components may be maintained separately as the project evolves.
