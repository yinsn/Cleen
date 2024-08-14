from typing import Optional

from pandas.core.frame import DataFrame

from .base import BasePreprocessorConfig, BasePreprocessor

class CoarsenPreprocessorConfig(BasePreprocessorConfig):
    treatment_column: Optional[str] = "treatment"
    

class CoarsenPreprocessor(BasePreprocessor):
    def __init__(self, dataframe: Optional[DataFrame] = None, config_path: Optional[str] = None) -> None:
        super().__init__(dataframe, config_path)
        self.treatment_column = self.config.treatment_column

    