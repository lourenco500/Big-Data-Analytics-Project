import pandas as pd
from pyod.models.iforest import IForest

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType as IT


class IForestOutlierRemover(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom MLlib Transformer that:
      1. Trains an Isolation Forest on a sample of the data (big-data-safe)
      2. Scores all partitions using a broadcast model
      3. Returns the DataFrame with outliers removed

    Parameters:
      - outlier_features:  list of column names to use for scoring
      - contamination:     proportion of outliers expected (default 0.05)
      - sample_fraction:   fraction of data to train on (default 0.1)
      - seed:              random seed for reproducibility (default 42)
    """

    outlier_features = Param(Params._dummy(), "outlier_features", "Feature columns for outlier detection")
    contamination    = Param(Params._dummy(), "contamination",    "Expected proportion of outliers")
    sample_fraction  = Param(Params._dummy(), "sample_fraction",  "Fraction of data to train on")
    seed             = Param(Params._dummy(), "seed",             "Random seed")

    def __init__(self, outlier_features=None, contamination=0.05, sample_fraction=0.1, seed=42):
        super().__init__()
        self._setDefault(
            outlier_features=outlier_features or [],
            contamination=contamination,
            sample_fraction=sample_fraction,
            seed=seed
        )

    def _transform(self, df: DataFrame) -> DataFrame:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        features      = self.getOrDefault("outlier_features")
        contamination = self.getOrDefault("contamination")
        fraction      = self.getOrDefault("sample_fraction")
        seed          = self.getOrDefault("seed")

        # 1. Train on a sample on the driver
        sample_pdf = df.sample(fraction=fraction, seed=seed).toPandas()
        clf = IForest(contamination=contamination, random_state=seed)
        clf.fit(sample_pdf[features].values)

        # 2. Broadcast fitted model to all executors
        clf_broadcast = spark.sparkContext.broadcast(clf)

        # 3. Build output schema
        scored_schema = StructType(
            df.schema.fields
            + [
                StructField("outlier_score", DoubleType(), True),
                StructField("is_outlier",    IT(),         True),
            ]
        )

        # 4. Scoring function — runs on executors, no re-fitting
        def score_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            model = clf_broadcast.value
            X = pdf[features].values
            pdf = pdf.copy()
            pdf["outlier_score"] = model.decision_function(X)
            pdf["is_outlier"]    = model.predict(X)
            return pdf

        # 5. Apply and filter
        return (
            df
            .groupby()
            .applyInPandas(score_partition, schema=scored_schema)
            .filter("is_outlier = 0")
            .drop("outlier_score", "is_outlier")
        )