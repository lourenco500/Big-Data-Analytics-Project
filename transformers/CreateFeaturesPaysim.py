from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, abs as spark_abs


class CreateFeaturesPaysim(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom MLlib Transformer that adds fraud-detection features to the PaySim dataset.
    
    Added columns:
      - error_balance_orig:  absolute discrepancy in sender balance after transaction
      - error_balance_dest:  absolute discrepancy in recipient balance after transaction
      - balance_ratio_orig:  transaction amount relative to sender's opening balance
    """

    def _transform(self, df: DataFrame) -> DataFrame:
        return (
            df
            .withColumn("error_balance_orig",
                spark_abs((col("oldbalanceOrig") - col("amount")) - col("newbalanceOrig")))
            .withColumn("error_balance_dest",
                spark_abs((col("oldbalanceDest") + col("amount")) - col("newbalanceDest")))
            .withColumn("balance_ratio_orig",
                col("amount") / (col("oldbalanceOrig") + 1))
        )