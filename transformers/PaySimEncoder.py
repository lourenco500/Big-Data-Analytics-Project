from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType as IT

class PaySimEncoder(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Casts boolean and target columns to IntegerType for VectorAssembler
    and Spark ML classifier compatibility. Pure Transformer — no fitting.
    """

    bool_cols = Param(Params._dummy(), "bool_cols", "Boolean columns to cast to integer")

    def __init__(self, bool_cols=None):
        super().__init__()
        self._setDefault(bool_cols=bool_cols or ["isFlaggedFraud"])

    def _transform(self, df: DataFrame) -> DataFrame:
        bool_cols = self.getOrDefault("bool_cols")
        for c in bool_cols:
            df = df.withColumn(c, col(c).cast(IT()))
        df = df.withColumn("isFraud", col("isFraud").cast(IT()))
        return df