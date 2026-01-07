# Machine Learning Zoomcamp (Cohort 2025)

## Capstone Project 1: A Genetic Variants Conflicting Classification Predictor Service

### Description of the problem

This project focuses on the classification of genetic variants using data from [ClinVar], a freely accessible, public archive hosted by the **National Center for Biotechnology Information (NCBI)**. This work has been inspired by previous research in the field by [Alexandra Veres].

#### What is ClinVar?

[ClinVar]: https://www.ncbi.nlm.nih.gov/clinvar/intro

According to the _NCBI_, **ClinVar** is a centralized repository that aggregates reports of the relationships among human variations and phenotypes, with supporting evidence. It facilitates access to and communication about the relationships asserted between human variation and observed health status. Interpretations of the clinical significance of variants are submitted by clinical testing laboratories, research laboratories, and expert panels worldwide.

#### Key Characteristics of the Dataset:

Based on the original study by Alexandra Veres, this dataset is characterized by:

- Real-world Genomic Data: Extracted from the NCBI ClinVar database.

- Feature Diversity: 65 initial features covering genomic coordinates, molecular consequences, and clinical metadata.

- Target Variable: A binary label (CLASS) where 1 indicates a conflict in clinical interpretation and 0 indicates consistency.

- Scientific Relevance: Ideal for testing classification models in a high-stakes healthcare context where data imbalance and feature selection are critical.

### Repository contents

This repository is a fork of the **`#mlzoomcamp`** course. The project was implemented at this folder: cohorts/2025/projects/midterm. This folder is organized to support the development, training, and deployment of the fraud detection model.

Below is an overview of the key directories and files:

- README.md: This file, providing an overview of the project.
- pyproject.toml: The `uv` managed project file.
- uv.lock: The real versions of the dependencies used here.
- train.py: The training script.
- predict.py: The predictive **FastAPI** application.
- pipeline_v1.bin: The serialized trained model.
- Dockerfile: The deployment docker file.
- run-docker.sh: A script to run "docker build" & "docker run" locally.
- submission.ipynb: The rest of the data cleaning, data analysis and predictive modelling stuff.
- clinvar_conflicting.csv.zip: A saved copy of the data set file downloaded from Kaggle.
- prepared_clinvar_conflicting.parquet: A local copy of the cleaned and prepared data set.

### How to run this project

This project was created using **Github's Codespaces**. To run any artifact create a free codespace from the repository's main page. Then from your browser: open a terminal view and change current directory to:

```
$ cd /workspaces/machine-learning-zoomcamp-2025-projects-capstone1
```

Now, from this directory, you can run:

- `$ uv run train.py` # to train the model with local data file
- `$ uv run predict.py` # to test the prediction microservice with local `pipeline_v1.bin`
- `$ ./run-docker.sh` # to create a docker image of the service and run it locally

To test the service, deployed at **Google Cloud Run** visit the documentation page at:

- <https://machine-learning-zoomcamp-428800185377.europe-west1.run.app>

Or run from any Python session (e.g. `uv run python`) the following code:

```python
import requests

variant = {
  "chrom":  "1",
  "pos":    1168180,
  "ref":    "G",
  "alt":    "C",
  "af_esp": 0.07710,
  "mc":     "SO:0001583|missense_variant",
  "origin": 1,
  "protein_position": 174.0,
  "amino_acids":      "E/D",
  "strand":           "1",
  "loftool":          0.157,
  "cadd_phred":       1.053,
  "cadd_raw":         -0.208682,
  "location_type":    "exon",
  "location_value":   "1/1",
}

res = requests.post("https://machine-learning-zoomcamp-428800185377.europe-west1.run.app/predict", json=transaction)
res.raise_for_status()

print(res.json())
```

More information about deployment can be found at [submission.ipynb].
