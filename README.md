# Big Data Analytics Project

**NOVA IMS · Spring 2026**

### Group 16 Members:

- Ana Margarida Macedo (20250405)
- Catarina Aboim (20250375)
- Margarida Craveiro (20250346)
- Matilde Simões (20250382)
- Lourenço Silva (20250453)

---

## Overview

This project demonstrates the use of **PySpark** for large-scale data processing, machine learning, graph analytics, and real-time streaming across three datasets. It is structured as a consulting prototype for an imaginary client, showing how Spark can solve data challenges that traditional tools cannot handle at scale.

---

## Datasets

| Dataset | Description | Used in |
|---|---|---|
| **NYC Yellow Taxi** | 12M+ trip records across Jan 2015 and Jan–Mar 2016. Includes fare, distance, pickup/dropoff zones, and rate codes. | Notebooks 1, 5 |
| **PaySim** | Synthetic financial transaction dataset simulating 30 days of mobile money activity. Severe class imbalance — fraud < 0.2% of transactions. | Notebooks 1, 2, 4 |
| **Fashion MNIST** | 70,000 grayscale images (28×28) of clothing items across 10 categories. | Notebook 3 |

---

## Project Structure

```
Big-Data-Analytics-Project/
│
├── Notebook1.ipynb                  # EDA & data cleaning (Taxi + PaySim)
├── Notebook2_mlPipelines.ipynb      # ML pipelines for fraud detection (PaySim)
├── Notebook3_DeepLearning.ipynb     # Deep learning for image classification (Fashion MNIST)
├── Notebook4_GraphFrames.ipynb      # Graph analytics on transaction network (PaySim)
├── Notebook5_Streaming.ipynb        # Real-time streaming pipeline (Taxi)
│
├── streaming/
│   ├── docker-compose.yml           # Kafka + Zookeeper cluster config
│   └── taxi_producer.py             # Kafka producer — replays taxi trips as live events
│
├── transformers/                    # Custom PySpark Transformer classes (used in Notebook 2)
│   ├── CreateFeaturesPaysim.py
│   ├── IForestOutlierRemover.py
│   ├── PaySimEncoder.py
│   └── ClassWeighter.py
│
├── data/
│   └── clean/                       # Parquet outputs from Notebook 1
│       ├── taxi_clean.parquet
│       ├── paysim_clean.parquet
│       ├── fashion_mnist_train_clean.parquet
│       └── fashion_mnist_test_clean.parquet
│
└── outputs/                         # Saved ML pipeline models
```

---

## Notebooks

### Notebook 1 — EDA & Data Cleaning
Covers the full data ingestion, profiling, and cleaning pipeline for both the NYC Taxi and PaySim datasets using **RDDs, DataFrames, and SparkSQL**. Flags and removes zero-distance trips, zero-fare records, negative amounts, and sub-60-second trips. Produces clean parquet files used by all downstream notebooks.

### Notebook 2 — Machine Learning Pipelines
Applies three classification models to the PaySim fraud detection problem: **Logistic Regression** (baseline), **Random Forest** with `CrossValidator` hyperparameter tuning, and **GBTClassifier**. All models are built as end-to-end Spark ML Pipelines using custom Transformer classes for feature engineering, outlier removal via Isolation Forest, encoding, and class weighting. Pipelines are saved to disk for reuse.

### Notebook 3 — Deep Learning
Applies deep learning to Fashion MNIST classification using two approaches: Spark MLlib's native **MultilayerPerceptronClassifier** with hyperparameter tuning, and **Transfer Learning** with a MobileNetV2 backbone (feature extraction via `predict_batch_udf`) combined with a PyTorch classification head trained via **TorchDistributor** for distributed training.

### Notebook 4 — GraphFrames
Models the PaySim transaction dataset as a directed graph (accounts as vertices, transactions as edges) and applies graph algorithms including **PageRank**, **Connected Components**, **BFS**, **Triangle Counting**, and **Motif Finding**. Compares the structural properties of fraudulent vs. legitimate accounts in the network.

### Notebook 5 — Spark Structured Streaming
Replays the cleaned taxi dataset as a live event stream through **Apache Kafka** and consumes it with **Spark Structured Streaming**. Implements windowed aggregations (3-minute tumbling windows per pickup zone), a stream-stream join correlating long-haul trips with high-fare activity, and a stream-static join enriching trips with zone metadata.

---

## Spark Features Demonstrated

| Feature | Notebook |
|---|---|
| RDDs | 1 |
| DataFrames & SparkSQL | 1, 2, 4 |
| ML Pipelines & custom Transformers | 2 |
| MLlib classification (LR, RF, GBT) | 2 |
| CrossValidator hyperparameter tuning | 2, 3 |
| Deep Learning (MLP, TorchDistributor) | 3 |
| Transfer Learning with predict_batch_udf | 3 |
| GraphFrames (PageRank, BFS, motifs) | 4 |
| Spark Structured Streaming + Kafka | 5 |
| Stream-stream joins with watermarks | 5 |
| Windowed aggregations | 5 |