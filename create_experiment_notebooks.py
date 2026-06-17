import json
import textwrap
from pathlib import Path


NOTEBOOKS = {
    "logistic_regression": {
        "filename": "01_experiment_logistic_regression.ipynb",
        "title": "Eksperimen Logistic Regression",
    },
    "random_forest": {
        "filename": "02_experiment_random_forest.ipynb",
        "title": "Eksperimen Random Forest",
    },
    "xgboost": {
        "filename": "03_experiment_xgboost.ipynb",
        "title": "Eksperimen XGBoost",
    },
}


def md(source):
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source):
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


COMMON_CELLS = [
    md(
        """
        ## Tujuan Notebook

        Notebook ini mereplikasi pipeline paper rujukan untuk ekstraksi diagram kelas UML dari teks kebutuhan.

        Tiga target/sub-model yang dilatih:

        - `class_attribute`: memprediksi `Tag` sebagai `Class`, `Attribute`, atau `Other`.
        - `class_attribute_relationship`: memprediksi `Class_Related`.
        - `class_class_relationship`: memprediksi `Class_R`.

        Transformasi fitur disamakan untuk seluruh model:

        - `Word`: `TfidfVectorizer(ngram_range=(1, 2))`.
        - `Sentence`: `TfidfVectorizer` unigram + `TruncatedSVD(n_components=100)`.
        - `POS`: `OneHotEncoder`.
        - Rasio train/test: 80/20.
        - Optimasi: `GridSearchCV` dengan 5-fold cross-validation.
        """
    ),
    code(
        """
        # Jalankan jika environment belum punya dependensi berikut.
        # %pip install pandas scikit-learn nltk joblib matplotlib
        # Khusus notebook XGBoost:
        # %pip install xgboost
        """
    ),
    code(
        """
        from pathlib import Path

        MODEL_NAME = "__MODEL_NAME__"
        RANDOM_STATE = 42
        TEST_SIZE = 0.20
        CV_FOLDS = 5
        SVD_COMPONENTS = 100

        DATA_DIR = Path("Dataset")
        OUTPUT_DIR = Path("experiment_outputs") / MODEL_NAME
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Model aktif: {MODEL_NAME}")
        print(f"Output directory: {OUTPUT_DIR.resolve()}")
        """
    ),
    code(
        """
        import re
        import string
        import warnings
        from collections import defaultdict

        import nltk
        import numpy as np
        import pandas as pd
        from joblib import dump
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        from sklearn.base import BaseEstimator, ClassifierMixin, clone
        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import GridSearchCV, KFold, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        warnings.filterwarnings("ignore", category=UserWarning)

        def download_nltk_resources():
            resources = [
                ("tokenizers/punkt", "punkt"),
                ("tokenizers/punkt_tab", "punkt_tab"),
                ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                ("corpora/wordnet", "wordnet"),
                ("corpora/omw-1.4", "omw-1.4"),
            ]
            for lookup_name, download_name in resources:
                try:
                    nltk.data.find(lookup_name)
                except (LookupError, OSError):
                    nltk.download(download_name, quiet=True)

        download_nltk_resources()
        lemmatizer = WordNetLemmatizer()
        """
    ),
    md(
        """
        ## 1. Load Dataset

        Dataset utama menggunakan `Dataset/Requirements.csv`.
        Dataset 12 problem komparatif menggunakan `Dataset/r12_problems.csv.csv`.
        """
    ),
    code(
        """
        def load_csv_with_fallback(*paths):
            for path in paths:
                if Path(path).exists():
                    print(f"Loading: {path}")
                    return pd.read_csv(path)
            raise FileNotFoundError(f"Tidak ada file yang ditemukan dari kandidat: {paths}")

        train_raw = load_csv_with_fallback(DATA_DIR / "Requirements.csv")
        r12_raw = load_csv_with_fallback(DATA_DIR / "r12_problems.csv.csv", DATA_DIR / "r12_problems.csv")

        display(train_raw.head(2))
        print(train_raw.shape)
        display(r12_raw.head(2))
        print(r12_raw.shape)
        """
    ),
    md(
        """
        ## 2. Preprocessing dan Token-Level Labeling

        Bagian ini mengikuti struktur notebook rujukan:

        - teks kebutuhan dipecah menjadi kalimat dan kata dengan NLTK,
        - setiap kata diberi POS tag,
        - kata dilemmatize,
        - label token dibentuk dari kolom `Class`, `Atributes`, dan `Relationship`.
        """
    ),
    code(
        """
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith("J"):
                return wordnet.ADJ
            if treebank_tag.startswith("V"):
                return wordnet.VERB
            if treebank_tag.startswith("N"):
                return wordnet.NOUN
            if treebank_tag.startswith("R"):
                return wordnet.ADV
            return wordnet.NOUN

        def normalize_label(value):
            value = str(value).strip().lower()
            return lemmatizer.lemmatize(value, pos="n")

        def extract_class_attribute_mapping(attribute_string):
            class_attribute_mapping = {}
            if pd.isna(attribute_string):
                return class_attribute_mapping

            class_attribute_groups = re.findall(r"(\\w+)\\s*\\[([^\\]]+)\\]", str(attribute_string))
            for class_name, attributes in class_attribute_groups:
                class_name_lem = normalize_label(class_name)
                attributes_list = [attr.strip() for attr in attributes.split(",")]
                attributes_lem = [normalize_label(attr) for attr in attributes_list]
                class_attribute_mapping[class_name_lem] = attributes_lem
            return class_attribute_mapping

        def parse_relationships(relationship_string):
            relationships = []
            if pd.isna(relationship_string):
                return relationships

            for rel in str(relationship_string).split(","):
                rel = rel.strip()
                if " and " in f" {rel.lower()} ":
                    parts = re.split(r"\\s+and\\s+", rel, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        relationships.append(tuple(normalize_label(part) for part in parts))
            return relationships

        def tag_problem_classes_and_attributes(problem_number, problem, class_attribute_mapping, class_list_lem, relationships, sentence_offset=0):
            problem_numbers = []
            sentence_numbers = []
            problems = []
            sent_list = []
            word_list = []
            pos_list = []
            tag_list = []
            class_related_list = []
            class_r_list = []

            sentences = nltk.sent_tokenize(str(problem))
            sentence_counter = sentence_offset

            for sentence in sentences:
                sentence_counter += 1
                words = nltk.word_tokenize(sentence)
                words = [word for word in words if word.lower() not in string.punctuation]
                pos_tags = nltk.pos_tag(words)

                lemmatized_words = []
                for word, pos in pos_tags:
                    wordnet_pos = get_wordnet_pos(pos)
                    lemmatized_words.append(lemmatizer.lemmatize(word.lower(), pos=wordnet_pos))

                lemmatized_sentence = " ".join(lemmatized_words)

                for word, lemmatized_word, pos in zip(words, lemmatized_words, [p for _, p in pos_tags]):
                    tag = "Other"
                    found_class = "Other"
                    found_relationship = "Other"

                    attribute_found = False
                    for class_name, attributes in class_attribute_mapping.items():
                        if lemmatized_word in attributes:
                            tag = "Attribute"
                            found_class = class_name
                            attribute_found = True
                            break

                    if not attribute_found and lemmatized_word in class_list_lem:
                        tag = "Class"
                        found_class = lemmatized_word
                        for rel in relationships:
                            if found_class in rel:
                                found_relationship = rel[1] if rel[0] == found_class else rel[0]
                                break

                    problem_numbers.append(problem_number)
                    sentence_numbers.append(f"Sentence: {sentence_counter}")
                    problems.append(problem)
                    sent_list.append(lemmatized_sentence)
                    word_list.append(lemmatized_word)
                    pos_list.append(pos)
                    tag_list.append(tag)
                    class_related_list.append(found_class)
                    class_r_list.append(found_relationship)

            tagged_df = pd.DataFrame({
                "Problem_Number": problem_numbers,
                "Sentence #": sentence_numbers,
                "Problem": problems,
                "Sentence": sent_list,
                "Word": word_list,
                "POS": pos_list,
                "Tag": tag_list,
                "Class_Related": class_related_list,
                "Class_R": class_r_list,
            })
            return tagged_df, sentence_counter

        def build_tagged_dataset(raw_df):
            tagged_data = []
            sentence_offset = 0

            for index, row in raw_df.iterrows():
                problem_number = index + 1
                problem_text = row["Problem"]
                attribute_string = row["Atributes"]
                relationship_string = row.get("Relationship", "")
                class_list_string = row["Class"]

                class_list = [cls.strip() for cls in str(class_list_string).split(",")]
                class_list_lem = [normalize_label(cls) for cls in class_list]
                class_attribute_mapping = extract_class_attribute_mapping(attribute_string)
                relationships = parse_relationships(relationship_string)

                tagged_df, sentence_offset = tag_problem_classes_and_attributes(
                    problem_number,
                    problem_text,
                    class_attribute_mapping,
                    class_list_lem,
                    relationships,
                    sentence_offset=sentence_offset,
                )
                tagged_data.append(tagged_df)

            final_df = pd.concat(tagged_data, ignore_index=True)
            final_df = final_df[[
                "Problem_Number", "Sentence #", "Problem", "Sentence", "Word", "POS",
                "Tag", "Class_Related", "Class_R"
            ]]
            final_df = final_df.fillna("Other")
            final_df[["Sentence", "Word", "POS", "Tag", "Class_Related", "Class_R"]] = (
                final_df[["Sentence", "Word", "POS", "Tag", "Class_Related", "Class_R"]].astype(str)
            )
            return final_df

        train_tagged = build_tagged_dataset(train_raw)
        r12_tagged = build_tagged_dataset(r12_raw)

        display(train_tagged.head(12))
        print(train_tagged[["Tag", "Class_Related", "Class_R"]].nunique())
        print(train_tagged["Tag"].value_counts())
        """
    ),
    md(
        """
        ## 3. Pipeline Fitur dan Konfigurasi Model
        """
    ),
    code(
        """
        def one_hot_encoder():
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore", sparse=True)

        def make_preprocessor(extra_columns=None):
            extra_columns = extra_columns or []
            transformers = [
                ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2)), "Word"),
                ("sentence_tfidf", Pipeline([
                    ("tfidf", TfidfVectorizer()),
                    ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)),
                ]), "Sentence"),
                ("pos_onehot", one_hot_encoder(), ["POS"]),
            ]

            for column in extra_columns:
                transformers.append((f"{column.lower()}_onehot", one_hot_encoder(), [column]))

            return ColumnTransformer(transformers=transformers, remainder="drop")

        class XGBLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
            def __init__(
                self,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=1.0,
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
            ):
                self.n_estimators = n_estimators
                self.max_depth = max_depth
                self.learning_rate = learning_rate
                self.subsample = subsample
                self.random_state = random_state
                self.tree_method = tree_method
                self.n_jobs = n_jobs

            def fit(self, X, y):
                from xgboost import XGBClassifier

                self.label_encoder_ = LabelEncoder()
                y_encoded = self.label_encoder_.fit_transform(y)
                self.classes_ = self.label_encoder_.classes_
                self.model_ = XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    tree_method=self.tree_method,
                    n_jobs=self.n_jobs,
                )
                self.model_.fit(X, y_encoded)
                return self

            def predict(self, X):
                y_encoded = self.model_.predict(X).astype(int)
                return self.label_encoder_.inverse_transform(y_encoded)

            def predict_proba(self, X):
                return self.model_.predict_proba(X)

        def get_model_and_grid(model_name):
            if model_name == "logistic_regression":
                classifier = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)
                param_grid = {
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__class_weight": [None, "balanced"],
                }
                return classifier, param_grid, False

            if model_name == "random_forest":
                classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
                param_grid = {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 20],
                    "classifier__min_samples_split": [2, 5],
                    "classifier__class_weight": [None, "balanced"],
                }
                return classifier, param_grid, False

            if model_name == "xgboost":
                try:
                    from xgboost import XGBClassifier
                except ImportError as exc:
                    raise ImportError("Install xgboost terlebih dahulu: %pip install xgboost") from exc

                classifier = XGBLabelEncodedClassifier(
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    n_jobs=-1,
                )
                param_grid = {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [3, 5],
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__subsample": [0.8, 1.0],
                }
                return classifier, param_grid, False

            raise ValueError(f"MODEL_NAME tidak dikenal: {model_name}")

        BASE_CLASSIFIER, PARAM_GRID, ENCODE_TARGET = get_model_and_grid(MODEL_NAME)
        CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        TASKS = {
            "class_attribute": {
                "target": "Tag",
                "features": ["Sentence", "Word", "POS"],
                "extra_columns": [],
            },
            "class_attribute_relationship": {
                "target": "Class_Related",
                "features": ["Sentence", "Word", "POS", "Tag"],
                "extra_columns": ["Tag"],
            },
            "class_class_relationship": {
                "target": "Class_R",
                "features": ["Sentence", "Word", "POS", "Tag", "Class_Related"],
                "extra_columns": ["Tag", "Class_Related"],
            },
        }

        print(BASE_CLASSIFIER)
        print(PARAM_GRID)
        """
    ),
    md(
        """
        ## 4. Training 3 Sub-Model dengan Split 80:20 dan GridSearchCV 5-Fold
        """
    ),
    code(
        """
        train_idx, test_idx = train_test_split(
            train_tagged.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        def make_pipeline(extra_columns):
            return Pipeline([
                ("preprocessor", make_preprocessor(extra_columns)),
                ("classifier", clone(BASE_CLASSIFIER)),
            ])

        def fit_task(task_name, spec):
            X = train_tagged[spec["features"]]
            y = train_tagged[spec["target"]].astype(str)

            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            pipeline = make_pipeline(spec["extra_columns"])
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=PARAM_GRID,
                scoring="f1_weighted",
                cv=CV,
                n_jobs=-1,
                refit=True,
                verbose=1,
            )

            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            print(f"\\n=== {task_name} ===")
            print("Best params:", grid.best_params_)
            print("Best CV f1_weighted:", grid.best_score_)
            print(classification_report(y_test, y_pred, zero_division=0))

            return {
                "task_name": task_name,
                "target": spec["target"],
                "features": spec["features"],
                "extra_columns": spec["extra_columns"],
                "best_estimator": grid.best_estimator_,
                "best_params": grid.best_params_,
                "best_cv_score": grid.best_score_,
                "label_encoder": None,
                "y_test": y_test,
                "y_pred": pd.Series(y_pred, index=y_test.index),
            }

        trained_tasks = {}
        for task_name, spec in TASKS.items():
            trained_tasks[task_name] = fit_task(task_name, spec)

        for task_name, bundle in trained_tasks.items():
            dump(bundle, OUTPUT_DIR / f"{MODEL_NAME}_{task_name}.joblib")

        print(f"Model tersimpan di: {OUTPUT_DIR.resolve()}")
        """
    ),
    md(
        """
        ## 5. Tabel Evaluasi Setara Table 3

        Baris sub-model memakai rata-rata berbobot (`weighted avg`) dari `classification_report`.
        """
    ),
    code(
        """
        def metrics_from_prediction(name, y_true, y_pred):
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            weighted = report["weighted avg"]
            return {
                "Models": name,
                "Precision": weighted["precision"],
                "Recall": weighted["recall"],
                "F1-score": weighted["f1-score"],
                "Accuracy": accuracy_score(y_true, y_pred),
                "Support": len(y_true),
            }

        rows = []
        for task_name, bundle in trained_tasks.items():
            rows.append(metrics_from_prediction(task_name, bundle["y_test"], bundle["y_pred"]))

        table3 = pd.DataFrame(rows)

        macro_row = {
            "Models": "Macro avg",
            "Precision": table3["Precision"].mean(),
            "Recall": table3["Recall"].mean(),
            "F1-score": table3["F1-score"].mean(),
            "Accuracy": np.nan,
            "Support": table3["Support"].sum(),
        }
        weighted_row = {
            "Models": "Weighted avg",
            "Precision": np.average(table3["Precision"], weights=table3["Support"]),
            "Recall": np.average(table3["Recall"], weights=table3["Support"]),
            "F1-score": np.average(table3["F1-score"], weights=table3["Support"]),
            "Accuracy": np.nan,
            "Support": table3["Support"].sum(),
        }

        table3 = pd.concat([table3, pd.DataFrame([macro_row, weighted_row])], ignore_index=True)
        table3_display = table3.drop(columns=["Support"]).copy()
        table3_display[["Precision", "Recall", "F1-score", "Accuracy"]] = table3_display[
            ["Precision", "Recall", "F1-score", "Accuracy"]
        ].round(4)

        svm_paper_table3 = pd.DataFrame([
            {"Models": "class_attribute", "SVM_Precision": 0.85, "SVM_Recall": 0.82, "SVM_F1-score": 0.83, "SVM_Accuracy": 0.92},
            {"Models": "class_attribute_relationship", "SVM_Precision": 0.97, "SVM_Recall": 0.66, "SVM_F1-score": 0.65, "SVM_Accuracy": 0.94},
            {"Models": "class_class_relationship", "SVM_Precision": 0.98, "SVM_Recall": 0.97, "SVM_F1-score": 0.97, "SVM_Accuracy": 0.97},
            {"Models": "Macro avg", "SVM_Precision": 0.87, "SVM_Recall": 0.74, "SVM_F1-score": 0.72, "SVM_Accuracy": np.nan},
            {"Models": "Weighted avg", "SVM_Precision": 0.92, "SVM_Recall": 0.92, "SVM_F1-score": 0.92, "SVM_Accuracy": np.nan},
        ])

        table3_comparison = table3_display.merge(svm_paper_table3, on="Models", how="left")

        display(table3_display)
        display(table3_comparison)
        table3_display.to_csv(OUTPUT_DIR / "table3_metrics.csv", index=False)
        table3_comparison.to_csv(OUTPUT_DIR / "table3_vs_svm_paper.csv", index=False)
        """
    ),
    md(
        """
        ## 6. Evaluasi 12 Problem Setara Table 4

        Evaluasi pada 12 problem memakai inferensi bertahap:

        1. Prediksi `Tag`.
        2. Prediksi `Class_Related` memakai `Tag` hasil prediksi.
        3. Prediksi `Class_R` memakai `Tag` dan `Class_Related` hasil prediksi.
        """
    ),
    code(
        """
        def predict_task(task_name, feature_df):
            bundle = trained_tasks[task_name]
            return bundle["best_estimator"].predict(feature_df[bundle["features"]])

        def predict_cascade(parsed_df):
            predicted = parsed_df[["Sentence", "Word", "POS"]].copy()
            predicted["Tag"] = predict_task("class_attribute", predicted)
            predicted["Class_Related"] = predict_task("class_attribute_relationship", predicted)
            predicted["Class_R"] = predict_task("class_class_relationship", predicted)
            predicted["Problem_Number"] = parsed_df["Problem_Number"].values
            predicted["Sentence #"] = parsed_df["Sentence #"].values
            return predicted

        def aggregate_problem_metrics(true_df, pred_df):
            metric_rows = []
            for target in ["Tag", "Class_Related", "Class_R"]:
                report = classification_report(
                    true_df[target].astype(str),
                    pred_df[target].astype(str),
                    output_dict=True,
                    zero_division=0,
                )
                metric_rows.append({
                    "precision": report["weighted avg"]["precision"],
                    "recall": report["weighted avg"]["recall"],
                    "f1": report["weighted avg"]["f1-score"],
                })
            metrics = pd.DataFrame(metric_rows).mean()
            return {
                "Precision (%)": round(metrics["precision"] * 100, 2),
                "Recall (%)": round(metrics["recall"] * 100, 2),
                "F1 (%)": round(metrics["f1"] * 100, 2),
            }

        paper_table4 = pd.DataFrame({
            "Requirements": [f"Problem {i}" for i in range(1, 13)],
            "SVM paper F1 (%)": [87, 83, 81, 81, 74, 95, 96, 73, 81, 91, 80, 98],
            "DoMoBOT F1 (%)": [71, 77, 67, 80, 82, 93, 80, 90, 94, 94, 95, 89],
            "SVM paper Precision (%)": [84, 77, 76, 76, 77, 93, 93, 76, 77, 90, 77, 97],
            "DoMoBOT Precision (%)": [75, 83, 75, 86, 87, 100, 80, 93, 96, 100, 100, 100],
            "SVM paper Recall (%)": [91, 93, 90, 92, 71, 98, 100, 79, 91, 92, 85, 99],
            "DoMoBOT Recall (%)": [67, 71, 60, 75, 78, 87, 80, 87, 92, 89, 91, 80],
        })

        r12_predictions = []
        table4_rows = []

        for problem_number in sorted(r12_tagged["Problem_Number"].unique()):
            true_problem = r12_tagged[r12_tagged["Problem_Number"] == problem_number].copy()
            pred_problem = predict_cascade(true_problem)
            r12_predictions.append(pred_problem)

            row = {"Requirements": f"Problem {problem_number}"}
            row.update(aggregate_problem_metrics(true_problem, pred_problem))
            table4_rows.append(row)

        r12_predictions = pd.concat(r12_predictions, ignore_index=True)
        table4_model = pd.DataFrame(table4_rows)
        table4_comparison = table4_model.merge(paper_table4, on="Requirements", how="left")

        display(table4_model)
        display(table4_comparison)

        r12_predictions.to_csv(OUTPUT_DIR / "r12_predictions.csv", index=False)
        table4_model.to_csv(OUTPUT_DIR / "table4_r12_model.csv", index=False)
        table4_comparison.to_csv(OUTPUT_DIR / "table4_r12_vs_svm_domobot.csv", index=False)
        """
    ),
    md(
        """
        ## 7. Ekstraksi PlantUML Sederhana

        Cell ini membuat representasi PlantUML dari prediksi token-level. Hasilnya dapat dipakai untuk visualisasi diagram kelas.
        """
    ),
    code(
        """
        def build_plantuml(pred_df):
            class_attributes = defaultdict(set)
            relationships = set()

            class_words = sorted(set(pred_df.loc[pred_df["Tag"] == "Class", "Word"].astype(str)))
            for cls in class_words:
                class_attributes[cls.capitalize()]

            attr_rows = pred_df[
                (pred_df["Tag"] == "Attribute") &
                (pred_df["Class_Related"].astype(str).str.lower() != "other")
            ]
            for _, row in attr_rows.iterrows():
                cls = str(row["Class_Related"]).capitalize()
                attr = str(row["Word"]).capitalize()
                class_attributes[cls].add(attr)

            rel_rows = pred_df[
                (pred_df["Tag"] == "Class") &
                (pred_df["Class_R"].astype(str).str.lower() != "other")
            ]
            for _, row in rel_rows.iterrows():
                source = str(row["Word"]).capitalize()
                target = str(row["Class_R"]).capitalize()
                if source and target and source != target:
                    relationships.add((source, target))

            lines = ["@startuml"]
            for cls, attrs in sorted(class_attributes.items()):
                lines.append(f"class {cls} {{")
                for attr in sorted(attrs):
                    lines.append(f"  {attr}")
                lines.append("}")

            for source, target in sorted(relationships):
                lines.append(f"{source} -- {target}")

            lines.append("@enduml")
            return "\\n".join(lines)

        plantuml_dir = OUTPUT_DIR / "plantuml"
        plantuml_dir.mkdir(exist_ok=True)

        for problem_number in sorted(r12_predictions["Problem_Number"].unique()):
            pred_problem = r12_predictions[r12_predictions["Problem_Number"] == problem_number]
            plantuml_text = build_plantuml(pred_problem)
            (plantuml_dir / f"problem_{problem_number}.puml").write_text(plantuml_text, encoding="utf-8")

        print(build_plantuml(r12_predictions[r12_predictions["Problem_Number"] == 1]))
        print(f"File PlantUML tersimpan di: {plantuml_dir.resolve()}")
        """
    ),
]


def build_notebook(model_name, title):
    cells = [md(f"# {title}")]
    for cell in COMMON_CELLS:
        cell_copy = json.loads(json.dumps(cell))
        cell_copy["source"] = [
            line.replace("__MODEL_NAME__", model_name)
            for line in cell_copy["source"]
        ]
        cells.append(cell_copy)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    root = Path(__file__).resolve().parent
    for model_name, config in NOTEBOOKS.items():
        notebook = build_notebook(model_name, config["title"])
        output_path = root / config["filename"]
        output_path.write_text(
            json.dumps(notebook, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Created {output_path}")


if __name__ == "__main__":
    main()
