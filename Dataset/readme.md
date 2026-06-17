# Requirements Datasets

This repository contains three CSV datasets focusing on requirement specifications and problem analysis for training and testing models. Each dataset serves a distinct purpose, providing structured data on requirement descriptions, problem domains, and related attributes.

## Datasets

### 1. `Requirements.csv`
This file contains 106 requirement questions, along with their associated classes, attributes, and relationships. This dataset is used for training models in the domain of requirement classifications and attribute extraction.

#### Columns:
- **Problem**: A textual description of the problem or scenario.
- **Class**: The main classes or entities involved.
- **Attributes**: The attributes associated with each class.
- **Relationship**: Relationships between the different classes.

#### Sample:
| Problem | Class | Attributes | Relationship |
|---------|-------|------------|--------------|
| Consider a movie database... | Movie, Studio, Genre... | Movie [Movie ID, Title...] | Movie and Studio... |

---

### 2. `Requirements_ProblemAnalysis.csv`
This file expands on `Requirements.csv` by adding textual analysis metadata for all 106 requirements, including information such as word counts, sentence counts, and domain classifications. 

#### Columns:
- **No**: Unique identifier for each requirement.
- **Requirements Text**: Full requirement text.
- **Class**: The classes/entities involved.
- **Attributes**: Attributes linked to each class.
- **Relationship**: Defined relationships between classes.
- **Source**: The origin of the requirement (e.g., Academic sources).
- **Domain**: The domain of the problem (e.g., Health, Business).
- **No. of Sentences**: Number of sentences in the requirement.
- **No. of Words**: Total number of words.
- **No. of Unique Words**: Count of unique words in the text.
- **No. of Words After Stop Words Elimination**: Word count after stop words are removed.
- **No. of Classes**: Number of classes involved.
- **No. of Attributes**: Number of attributes mentioned.
- **No. of Relationships**: Number of relationships between classes.

#### Sample:
| No | Requirements Text | Class | Attributes | Relationship | Source | Domain | No. of Sentences |
|----|-------------------|-------|------------|--------------|--------|--------|------------------|
| R1 | Consider a movie... | Movie, Studio... | Movie [Movie ID, ...] | Movie and Studio... | Academic | Entertainment | 14 |

---

### 3. `r12_problems.csv`
This file contains a subset of 12 requirement problems sourced from Domobot, detailing their class and attribute definitions. It is designed for testing and evaluating models on a smaller dataset of problems.

#### Columns:
- **Problem**: Description of the requirement problem.
- **Class**: The classes or entities involved.
- **Attributes**: The attributes associated with each class.
- **Relationship**: Relationships between the different classes.

#### Sample:
| Problem | Class | Attributes | Relationship |
|---------|-------|------------|--------------|
| A company is comprised... | Company, Department... | Department [ID, ...] | Company and Department... |

---

## Usage

These datasets can be used for:
- **Training machine learning models** for text classification, attribute extraction, and relationship identification.
- **Benchmarking algorithms** designed for requirement parsing and domain-specific problem analysis.
- **Textual analysis** of problem structures across various domains.



