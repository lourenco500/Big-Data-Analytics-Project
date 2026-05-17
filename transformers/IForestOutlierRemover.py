import pandas as pd
from pyod.models.iforest import IForest

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType, IntegerType as IT


class IForestOutlierRemover(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom MLlib Transformer that:
      1. Trains an Isolation Forest on a sample of the data (big-data-safe)
      2. Scores all partitions using a broadcast model via two scalar pandas_udfs
      3. Returns the DataFrame with outliers removed

    Implementation note: we use two scalar pandas_udfs (one per output column)
    rather than `applyInPandas`. This avoids the StructArray.from_arrays bug in
    PyArrow >= 14 that breaks the applyInPandas → Arrow round-trip.

    Parameters:
      - outlier_features:  list of column names to use for scoring
      - contamination:     proportion of outliers expected (default 0.05;
                           set to 0 to disable removal effectively)
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
            seed=seed,
        )

    def _transform(self, df: DataFrame) -> DataFrame:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        features      = self.getOrDefault("outlier_features")
        contamination = self.getOrDefault("contamination")
        fraction      = self.getOrDefault("sample_fraction")
        seed          = self.getOrDefault("seed")

        # Short-circuit when contamination is 0 — no rows would be flagged,
        # so skip the expensive sample-fit-broadcast-score scan entirely.
        if contamination == 0:
            return df

        # 1. Train on a driver-side sample
        sample_pdf = df.sample(fraction=fraction, seed=seed).toPandas()
        clf = IForest(contamination=contamination, random_state=seed)
        clf.fit(sample_pdf[features].values)

        # 2. Broadcast fitted model to executors
        clf_broadcast = spark.sparkContext.broadcast(clf)

        # 3. Scalar pandas_udfs — each returns a single Series, no StructArray
        #    construction (which is what trips PyArrow >= 14 in applyInPandas).
        @pandas_udf(DoubleType())
        def outlier_score_udf(*cols: pd.Series) -> pd.Series:
            model = clf_broadcast.value
            X = pd.concat(cols, axis=1).to_numpy()
            return pd.Series(model.decision_function(X).astype("float64"))

        @pandas_udf(IT())
        def is_outlier_udf(*cols: pd.Series) -> pd.Series:
            model = clf_broadcast.value
            X = pd.concat(cols, axis=1).to_numpy()
            return pd.Series(model.predict(X).astype("int32"))

        # 4. Apply as new columns, filter, drop scoring columns
        feature_cols = [col(c) for c in features]
        return (
            df
            .withColumn("outlier_score", outlier_score_udf(*feature_cols))
            .withColumn("is_outlier",    is_outlier_udf(*feature_cols))
            .filter("is_outlier = 0")
            .drop("outlier_score", "is_outlier")
        )
