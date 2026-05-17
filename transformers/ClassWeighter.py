from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit


class ClassWeighter(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Adds a `class_weight` column with weights inversely proportional to class
    frequency in the DataFrame. Used as `weightCol=` in Spark ML classifiers
    that support sample weighting (LogisticRegression, RandomForestClassifier,
    GBTClassifier from Spark 3.0+).

    Weight formula per class c:
        weight_c = total_rows / (n_classes * count_of_class_c)

    This is the same scheme as scikit-learn's `class_weight="balanced"`.
    On PaySim (where fraud is ~0.13% of rows), positive examples receive a
    much larger weight than negatives, forcing the classifier to take them
    seriously instead of predicting "not fraud" for everything.

    Parameters:
      - label_col:   name of the label column (default "isFraud")
      - output_col:  name of the weight column to add (default "class_weight")

    Note: this is a pure Transformer — it recomputes the weights from
    whatever DataFrame it sees. Since `weightCol` is only consumed at
    classifier fit-time, recomputing at test-time is harmless (the test
    weights are never used to update the model).
    """

    label_col  = Param(Params._dummy(), "label_col",  "Name of the label column")
    output_col = Param(Params._dummy(), "output_col", "Name of the weight column")

    def __init__(self, label_col="isFraud", output_col="class_weight"):
        super().__init__()
        self._setDefault(label_col=label_col, output_col=output_col)

    def _transform(self, df: DataFrame) -> DataFrame:
        label_col  = self.getOrDefault("label_col")
        output_col = self.getOrDefault("output_col")

        # Count rows per class — small result, safe to collect
        counts = df.groupBy(label_col).count().collect()
        total      = sum(r["count"] for r in counts)
        n_classes  = len(counts)

        # Inverse-frequency weights, matching sklearn's "balanced" scheme
        weights = {r[label_col]: total / (n_classes * r["count"]) for r in counts}

        # Build a chained when/otherwise expression
        expr = None
        for cls, w in weights.items():
            cond = (col(label_col) == lit(cls))
            expr = when(cond, lit(w)) if expr is None else expr.when(cond, lit(w))

        return df.withColumn(output_col, expr.otherwise(lit(1.0)))
