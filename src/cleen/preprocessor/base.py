from abc import ABCMeta
from typing import List, Optional

import pandas as pd
from paradance import CSVLoader, load_config
from pydantic import BaseModel


class BasePreprocessorConfig(BaseModel):
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_type: str = "csv"
    max_rows: Optional[int] = None
    group_by: Optional[str] = None
    categorical_columns: Optional[List[str]] = None


class BasePreprocessor(metaclass=ABCMeta):
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.config = load_config(config_path)
        if dataframe is not None:
            self.dataframe = dataframe
        else:
            self._load_data()

    def _load_data(self) -> None:
        self.dataframe = CSVLoader(
            file_path=self.config.file_path,
            file_name=self.config.file_name,
            file_type=self.config.file_type,
            max_rows=self.config.max_rows,
        ).df
