import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

# TODO: mask - draw picture
# import matplotlib.pyplot as plt


class CEMMatcher:
    def __init__(
        self,
        dataframe: pd.DataFrame,  # the provided dataframe
        feature_columns: List[str],  # the columns of features to be used for matching
        label_column: str = "treatment",  # the label column, default is "treatment"
        index_column: str = "user_id",  # the index column, default is "user_id"
        is_key_to_key: bool = False,  # whether the matching is key-to-key, default is False
        coarsen_method: str = "sturges",  # the method to coarsen the features, default is "sturges"
    ):
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError('"dataframe" Must be Provided and Not Empty')
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            raise ValueError('"feature_columns" Must be Provided and Not Empty')
        for column in feature_columns:
            if column not in dataframe.columns:
                raise ValueError(f'"{column}" Not Found in "dataframe"')
        if label_column not in dataframe.columns:
            raise ValueError(f'"{label_column}" Not Found in "dataframe"')
        if index_column not in dataframe.columns:
            raise ValueError(f'"{index_column}" Not Found in "dataframe"')
        if coarsen_method not in ["sturges", "fd", "scott"]:
            raise NotImplementedError(
                f'Coarsen Method "{coarsen_method}" Not Implemented'
            )
        if dataframe.duplicated().sum() > 0:
            raise ValueError('Duplicate Rows in "dataframe"')
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError('Duplicate Rows in "feature_columns"')
        unique_labels = set(dataframe[label_column])
        if len(unique_labels) != 2:
            raise ValueError(
                f'"label_column" Column [{label_column}] Must Contain Two Distinct Values, Expected to be 0 and 1'
            )
        elif (0 not in unique_labels) or (1 not in unique_labels):
            new_label_mapping = {unique_labels.pop(): 0, unique_labels.pop(): 1}
            dataframe[label_column] = dataframe[label_column].map(new_label_mapping)
            warnings.warn(
                f'Values of "label_column" Column [{label_column}] are Expected to be 0 and 1. Values have been Adjusted to 0 and 1 Automaticly.',
                UserWarning,
            )
        if not dataframe[index_column].is_unique:
            raise ValueError(
                f'"index_column" Column [{index_column}] Must Have Unique Values'
            )
        self.dataframe = dataframe
        self.label_column = label_column
        self.index_column = index_column
        self.feature_columns = feature_columns
        self.is_key_to_key = is_key_to_key
        self.coarsen_method = coarsen_method
        # TODO: hard code
        self.feature_columns_bins = {}
        for feature_column in feature_columns:
            if f"{feature_column}_bins" not in self.dataframe.columns:
                self.feature_columns_bins[feature_column] = f"{feature_column}_bins"
                self.dataframe.insert(
                    len(self.dataframe.columns), f"{feature_column}_bins", None
                )
            else:
                i = 0
                while f"{feature_column}_{i}_bins" in self.dataframe.columns:
                    i += 1
                self.feature_columns_bins[feature_column] = f"{feature_column}_{i}_bins"
                self.dataframe.insert(
                    len(self.dataframe.columns), f"{feature_column}_{i}_bins", None
                )
        if "coarsen_group_index" not in self.dataframe.columns:
            self.coarsen_group_index_name = "coarsen_group_index"
            self.dataframe.insert(
                len(self.dataframe.columns), "coarsen_group_index", None
            )
        else:
            i = 0
            while f"coarsen_group_index_{i}" in self.dataframe.columns:
                i += 1
            self.coarsen_group_index_name = f"coarsen_group_index_{i}"
            self.dataframe.insert(
                len(self.dataframe.columns), f"coarsen_group_index_{i}", None
            )
        if "matching_weights" not in self.dataframe.columns:
            self.matching_weights_column = "matching_weights"
            self.dataframe.insert(len(self.dataframe.columns), "matching_weights", None)
        else:
            i = 0
            while f"matching_weights_{i}" in self.dataframe.columns:
                i += 1
            self.matching_weights_column = f"matching_weights_{i}"
            self.dataframe.insert(
                len(self.dataframe.columns), f"matching_weights_{i}", None
            )

    def _coarsen_features(self) -> None:
        """
        Coarsen Features by Creating New Columns, to be used for Exact Matching.
        """
        # TODO: descrete and not number feature
        if self.coarsen_method == "sturges":
            for feature_column in self.feature_columns:
                bins = int(np.ceil(np.log2(self.dataframe[feature_column].count()) + 1))
                self.dataframe[self.feature_columns_bins[feature_column]] = pd.cut(
                    self.dataframe[feature_column], bins=bins, labels=False
                )
            coarsen_features_values = (
                self.dataframe[
                    [
                        self.feature_columns_bins[feature_column]
                        for feature_column in self.feature_columns
                    ]
                ]
                .astype(str)
                .agg("_".join, axis=1)
            )
            self.dataframe[self.coarsen_group_index_name] = coarsen_features_values
        elif self.coarsen_method == "fd":
            for feature_column in self.feature_columns:
                iqr = stats.iqr(self.dataframe[feature_column])
                if iqr == 0:
                    var = np.var(self.dataframe[feature_column])
                    width = 3.5 * np.sqrt(var)
                else:
                    width = 2 * iqr
                min_value = self.dataframe[feature_column].min()
                max_value = self.dataframe[feature_column].max()
                bins = int(
                    max(
                        1,
                        np.ceil(
                            (max_value - min_value)
                            / width
                            * self.dataframe[feature_column].shape[0] ** (1 / 3)
                        ),
                    )
                )
                self.dataframe[self.feature_columns_bins[feature_column]] = pd.cut(
                    self.dataframe[feature_column], bins=bins, labels=False
                )
            coarsen_features_values = (
                self.dataframe[
                    [
                        self.feature_columns_bins[feature_column]
                        for feature_column in self.feature_columns
                    ]
                ]
                .astype(str)
                .agg("_".join, axis=1)
            )
            self.dataframe[self.coarsen_group_index_name] = coarsen_features_values
        elif self.coarsen_method == "scott":
            for feature_column in self.feature_columns:
                width = (
                    3.5
                    * np.sqrt(np.var(self.dataframe[feature_column]))
                    * self.dataframe[feature_column].count() ** (-1 / 3)
                )
                min_value = self.dataframe[feature_column].min()
                max_value = self.dataframe[feature_column].max()
                bins = int(max(1, np.ceil((max_value - min_value) / width)))
                self.dataframe[self.feature_columns_bins[feature_column]] = pd.cut(
                    self.dataframe[feature_column], bins=bins, labels=False
                )
            coarsen_features_values = (
                self.dataframe[
                    [
                        self.feature_columns_bins[feature_column]
                        for feature_column in self.feature_columns
                    ]
                ]
                .astype(str)
                .agg("_".join, axis=1)
            )
            self.dataframe[self.coarsen_group_index_name] = coarsen_features_values
        else:
            raise NotImplementedError(
                f'Coarsen Method "{self.coarsen_method}" Not Implemented'
            )
        # TODO: mask - draw picture
        # for feature_column in self.feature_columns:
        #     bin_counts = self.dataframe[self.feature_columns_bins[feature_column]].value_counts().sort_index()
        #     plt.figure(figsize=(10, 6))
        #     plt.bar(bin_counts.index, bin_counts.values, edgecolor="black")
        #     plt.title(f"Feature {feature_column} Distribution")
        #     plt.xlabel("Bins")
        #     plt.ylabel("Frequency")
        #     plt.xticks(bin_counts.index)
        #     plt.grid(True)
        #     plt.savefig(f"{feature_column}_bins.png", dpi=300)
        #     plt.close()

    def _exact_matching(self) -> pd.DataFrame:
        """
        Exact Matching based on the Result of _coarsen_features(). Return the Matched DataFrame with Extra Information.
        """
        matching_result_index = []
        for group_name, group in self.dataframe.groupby(self.coarsen_group_index_name):
            unique_labels = group[self.label_column].unique()
            if (0 in unique_labels) and (1 in unique_labels):
                if self.is_key_to_key:
                    if (
                        group[group[self.label_column] == 0].count()
                        >= group[group[self.label_column] == 1].count()
                    ):
                        matching_result_index.extend(
                            group[group[self.label_column] == 1][self.index_column]
                        )
                        matching_result_index.extend(
                            group[group[self.label_column] == 0][
                                self.index_column
                            ].sample(group[group[self.label_column] == 1].count())
                        )
                    else:
                        matching_result_index.extend(
                            group[group[self.label_column] == 0][self.index_column]
                        )
                        matching_result_index.extend(
                            group[group[self.label_column] == 1][
                                self.index_column
                            ].sample(group[group[self.label_column] == 1].count())
                        )
                else:
                    matching_result_index.extend(group[self.index_column])
                    group_matching_weight = float(
                        group[group[self.label_column] == 1][self.index_column].count()
                    ) / float(
                        group[group[self.label_column] == 0][self.index_column].count()
                    )
                    data_in_group = self.dataframe[self.index_column].isin(
                        group[self.index_column]
                    )
                    self.dataframe.loc[data_in_group, self.matching_weights_column] = (
                        group_matching_weight
                    )
        dataframe_matched = self.dataframe[
            self.dataframe[self.index_column].isin(matching_result_index)
        ]
        if self.is_key_to_key:
            dataframe_matched[self.matching_weights_column] = 1.0
        else:
            total_matching_weight = float(
                dataframe_matched[dataframe_matched[self.label_column] == 0][
                    self.index_column
                ].count()
            ) / float(
                dataframe_matched[dataframe_matched[self.label_column] == 1][
                    self.index_column
                ].count()
            )
            dataframe_matched.loc[
                :, self.matching_weights_column
            ] *= total_matching_weight
            treament_data = dataframe_matched[self.label_column] == 1
            dataframe_matched.loc[treament_data, self.matching_weights_column] = 1.0
        return dataframe_matched

    def match(self) -> pd.DataFrame:
        """
        Call _coarsen_features() and _exact_matching() to Perform the Matching. Return the Matched DataFrame.
        """
        self._coarsen_features()
        dataframe_matched_raw = self._exact_matching()
        dataframe_matched = dataframe_matched_raw.copy()
        for feature_column in self.feature_columns:
            dataframe_matched.drop(
                self.feature_columns_bins[feature_column], axis=1, inplace=True
            )
        dataframe_matched.drop(self.coarsen_group_index_name, axis=1, inplace=True)
        return dataframe_matched
