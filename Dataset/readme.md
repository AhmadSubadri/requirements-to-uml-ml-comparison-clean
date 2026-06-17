# Dataset Documentation

This folder contains the datasets used by the requirements-to-UML class diagram extraction experiments.

## Files

```text
Requirements.csv
Requirements_ProblemAnalysis.csv
r12_problems.csv.csv
real_world_studies.csv
real_world_studies_template.csv
```

## 1. `Requirements.csv`

Main dataset used for training and internal 80:20 token-level evaluation.

### Columns

| Column | Description |
|---|---|
| `Problem` | Natural language software requirement description. |
| `Class` | Comma-separated class/entity names. |
| `Atributes` | Class-attribute annotations in the format `Class [attr1, attr2]`. The column name follows the original dataset spelling. |
| `Relationship` | Class-class relationships in the format `ClassA and ClassB`. |

### Usage

Used by:

- `01_experiment_logistic_regression.ipynb`
- `02_experiment_random_forest.ipynb`
- `03_experiment_xgboost.ipynb`

The notebooks convert this dataset into token-level labels:

- `Tag`
- `Class_Related`
- `Class_R`

## 2. `Requirements_ProblemAnalysis.csv`

Metadata and statistical overview for the main requirement problems.

### Columns

| Column | Description |
|---|---|
| `No` | Requirement identifier. |
| `Requirements Text` | Full requirement text. |
| `Class` | Annotated classes/entities. |
| `Atributes` | Annotated attributes. |
| `Relationship` | Annotated relationships. |
| `Source` | Requirement source. |
| `Domain` | Application domain. |
| `No. of Sentences` | Number of sentences. |
| `No. of Words` | Number of words. |
| `No. of Unique Words` | Number of unique words. |
| `No. of Words After Stop Words Elimination` | Word count after stop-word removal. |
| `No. of Classes` | Number of annotated classes. |
| `No. of Attributes` | Number of annotated attributes. |
| `No. of Relationships` | Number of annotated relationships. |

### Usage

Used for dataset description and Table 1-style statistics in the paper/report.

## 3. `r12_problems.csv.csv`

Subset of 12 requirement problems used for comparative evaluation against the reference SVM paper and DoMoBOT.

### Columns

| Column | Description |
|---|---|
| `Problem` | Natural language requirement problem. |
| `Class` | Annotated classes/entities. |
| `Atributes` | Annotated attributes. |
| `Relationship` | Annotated class-class relationships. |

### Usage

Used by notebooks 01-03 to generate Table 4-style evaluation results.

## 4. `real_world_studies.csv`

Reconstructed real-world case-study dataset used by:

```text
04_experiment_real_world_studies.ipynb
```

It contains two case studies:

| System | Requirements name | Domain |
|---|---|---|
| System 1 | Stroke recovery assistant | Health |
| System 2 | Archive space project | Information system |

### Columns

| Column | Description |
|---|---|
| `System` | System identifier, e.g. `System 1`. |
| `Requirements name` | Case-study name. |
| `Domain` | Case-study domain. |
| `Problem` | Reconstructed natural language requirement text. |
| `Class` | Manual reference class annotation. |
| `Atributes` | Manual reference class-attribute annotation. |
| `Relationship` | Manual reference class-class relationship annotation. |

### Important Validity Note

The full original texts and expert annotations for the real-world studies were not explicitly available in the public reference repository or in the reference paper text. Therefore, this file contains reconstructed case studies based on the reported system names, domains, class counts, attribute counts, relationship counts, and conceptual hints from the reference paper.

Use this dataset as an additional reconstructed case-study evaluation, not as an exact reproduction of the reference paper's real-world study.

## 5. `real_world_studies_template.csv`

Blank template for creating or replacing real-world case-study inputs.

Use this file if you want to add manually verified case studies. Copy it to:

```text
real_world_studies.csv
```

Then fill in all required columns before running notebook 04.

## Citation and Attribution

The main dataset and the 12-problem benchmark are credited to the original AutomatedRE work by Umar et al. and its public repository. Please cite the reference paper when using these datasets or baseline values.

