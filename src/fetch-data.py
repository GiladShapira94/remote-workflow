
import pandas as pd
import mlrun
from mlrun.artifacts import Artifact


def fetch_data(context, dataset: pd.DataFrame,format="csv"):
    context.logger.info("saving dataframe")
    context.log_dataset("dataset", df=dataset, format=format, index=False)
