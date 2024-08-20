from typing import List

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleen import CEMMatcher


def generate_random_test_dataframe(
    num_features: int,  # the number of features column in the dataframe
    num_samples: int,  # the number of samples in the dataframe
    treatment_ratio: float = 0.5,  # the ratio of treatment samples in the dataframe, default is 0.5
    label_column: str = "treatment",  # the label column, default is "treatment"
    index_column: str = "user_id",  # the index column, default is "user_id"
) -> dict:
    """
    Generate a Random test Dataframe with the given Infomation.
    """
    feature_values = []
    for _ in range(num_features):
        feature_values.append(np.random.rand(num_samples))
    binary_labels = np.random.binomial(1, treatment_ratio, size=num_samples)
    user_ids = list(range(1, num_samples + 1))
    data = {}
    features = []
    for f in range(num_features):
        data[f"feature{f+1}"] = feature_values[f]
        features.append(f"feature{f+1}")
    data[label_column] = binary_labels
    data[index_column] = user_ids
    data_result = {"dataframe": pd.DataFrame(data), "features": features}
    return data_result


def cem_match(
    dataframe: pd.DataFrame,  # the provided dataframe
    feature_columns: List[str],  # the feature columns to be considered
    label_column: str = "treatment",  # the label column, default is "treatment"
    index_column: str = "user_id",  # the index column, default is "user_id"
    is_key_to_key: bool = False,  # whether the matching is key-to-key, default is False
    coarsen_method: str = "sturges",  # the method to coarsen the features, default is "sturges"
) -> dict:
    """
    Match the given Dataframe using CEM.
    """
    cem_matcher = CEMMatcher(
        dataframe,
        feature_columns,
        label_column,
        index_column,
        is_key_to_key,
        coarsen_method,
    )
    matched_dataframe = cem_matcher.match()
    cem_match_result = {
        "matched_dataframe": matched_dataframe,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "weight_column": matched_dataframe.columns[-1],
    }
    return cem_match_result


def balance_check(
    dataframe: pd.DataFrame,  # the dataframe to be checked for balance
    feature_columns: List[str],  # the feature columns to be considered
    label_column: str = "treatment",  # the label column, default is "treatment"
    weight_column: str = "matching_weights",  # the weight column to be used, default is "matching_weights"
) -> pd.DataFrame:
    """
    Check if the given Dataframe is Balanced for the given Feature Columns and Label Column.
    """
    # TODO: add wasserstein distance for balance check - stats.wasserstein_distance()
    balance_check_results = pd.DataFrame(
        columns=["features", "mean_difference", "t_statistics", "p_value"]
    )
    if weight_column not in dataframe.columns:
        dataframe.insert(len(dataframe.columns), weight_column, 1)
    for feature_column in feature_columns:
        weighted_mean_label_1 = np.average(
            dataframe[dataframe[label_column] == 1][feature_column],
            weights=dataframe[dataframe[label_column] == 1][weight_column],
        )
        weighted_mean_label_0 = np.average(
            dataframe[dataframe[label_column] == 0][feature_column],
            weights=dataframe[dataframe[label_column] == 0][weight_column],
        )
        mean_difference = abs(weighted_mean_label_1 - weighted_mean_label_0)
        # TODO: calc weighted_standard_deviation_label_1 and weighted_standard_deviation_label_0
        weighted_standard_deviation_label_1 = np.sqrt(
            np.average(
                (
                    dataframe[dataframe[label_column] == 1][feature_column]
                    - weighted_mean_label_1
                )
                ** 2,
                weights=dataframe[dataframe[label_column] == 1][weight_column],
            )
        )
        weighted_standard_deviation_label_0 = np.sqrt(
            np.average(
                (
                    dataframe[dataframe[label_column] == 0][feature_column]
                    - weighted_mean_label_0
                )
                ** 2,
                weights=dataframe[dataframe[label_column] == 0][weight_column],
            )
        )
        t_statistics, p_value = stats.ttest_ind_from_stats(
            weighted_mean_label_1,
            weighted_standard_deviation_label_1,
            len(dataframe[dataframe[label_column] == 1]),
            weighted_mean_label_0,
            weighted_standard_deviation_label_0,
            len(dataframe[dataframe[label_column] == 0]),
            equal_var=False,
        )
        feature_balance_check_result = {
            "features": feature_column,
            "mean_difference": mean_difference,
            "t_statistics": t_statistics,
            "p_value": p_value,
        }
        if balance_check_results.empty:
            balance_check_results = pd.DataFrame(
                feature_balance_check_result, index=[0]
            )
        else:
            balance_check_results = pd.concat(
                [
                    balance_check_results,
                    pd.DataFrame(feature_balance_check_result, index=[0]),
                ]
            )
    return balance_check_results


@pytest.mark.parametrize("num_features, num_samples, treatment_ratio", [(3, 2000, 0.6)])
def test_cem_balance_simple(
    num_features: int, num_samples: int, treatment_ratio: float
) -> None:
    """
    Test CEM Balance for the given test data.
    """
    if treatment_ratio > 1 or treatment_ratio < 0:
        pytest.skip("treatment_ratio should be between 0 and 1")
    test_raw_data = generate_random_test_dataframe(
        num_features, num_samples, treatment_ratio
    )
    matched_data = cem_match(test_raw_data["dataframe"], test_raw_data["features"])
    if matched_data["matched_dataframe"].empty:
        pytest.skip("No Matched Data Found")
    balance_results = balance_check(
        matched_data["matched_dataframe"],
        matched_data["feature_columns"],
        matched_data["label_column"],
        matched_data["weight_column"],
    )
    not_balanced_features = []
    is_balanced = True
    for _, row in balance_results.iterrows():
        if row["p_value"] < 0.05 or row["mean_difference"] > 0.1:
            not_balanced_features.append(row["features"])
            is_balanced = False
    assert is_balanced, f"Features {not_balanced_features} did not achieve balance"


def test_empty_dataframe() -> None:
    """
    Test CEM with an empty dataframe.
    """
    with pytest.raises(ValueError, match='"dataframe" Must be Provided and Not Empty'):
        CEMMatcher(pd.DataFrame(), [])


def test_invalid_data_type() -> None:
    """
    Test CEM with an invalid type of data.
    """
    with pytest.raises(ValueError, match='"dataframe" Must be Provided and Not Empty'):
        CEMMatcher("not_a_dataframe", [])


def test_empty_feature_list() -> None:
    """
    Test CEM with an empty feature list.
    """
    dataframe = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(
        ValueError, match='"feature_columns" Must be Provided and Not Empty'
    ):
        CEMMatcher(dataframe, [])


def test_missing_feature_column() -> None:
    """
    Test CEM with a missing feature column.
    """
    dataframe = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match='"b" Not Found in "dataframe"'):
        CEMMatcher(dataframe, ["a", "b"])


def test_missing_label_column() -> None:
    """
    Test CEM with a missing label column.
    """
    dataframe = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match='"treatment" Not Found in "dataframe"'):
        CEMMatcher(dataframe, ["a"], label_column="treatment")


def test_missing_index_column() -> None:
    """
    Test CEM with a missing index column.
    """
    dataframe = pd.DataFrame({"a": [1, 2, 3], "treatment": [1, 1, 0]})
    with pytest.raises(ValueError, match='"user_id" Not Found in "dataframe"'):
        CEMMatcher(dataframe, ["a"], index_column="user_id")


def test_invalid_coarsen_method() -> None:
    """
    Test CEM with an invalid coarsen method.
    """
    dataframe = pd.DataFrame(
        {"a": [1, 2, 3], "user_id": [1, 2, 3], "treatment": [1, 1, 0]}
    )
    with pytest.raises(
        NotImplementedError, match='Coarsen Method "invalid_method" Not Implemented'
    ):
        CEMMatcher(dataframe, ["a"], coarsen_method="invalid_method")


def test_duplicate_feature_columns() -> None:
    """
    Test CEM with duplicate feature columns.
    """
    dataframe = pd.DataFrame(
        {"a": [1, 2, 3], "user_id": [1, 2, 3], "treatment": [1, 1, 0]}
    )
    with pytest.raises(ValueError, match='Duplicate Features in "feature_columns"'):
        CEMMatcher(dataframe, ["a", "a"])


def test_non_binary_label() -> None:
    """
    Test CEM with a non-binary label.
    """
    dataframe = pd.DataFrame(
        {"a": [1, 2, 3], "user_id": [1, 2, 3], "treatment": [1, 2, 0]}
    )
    with pytest.raises(
        ValueError, match="Must Contain Two Distinct Values, Expected to be 0 and 1"
    ):
        CEMMatcher(dataframe, ["a"])


def test_label_mapping_warning() -> None:
    """
    Test CEM with a non-01 label.
    """
    dataframe = pd.DataFrame(
        {"a": [1, 2, 3], "user_id": [1, 2, 3], "treatment": [1, 1, 2]}
    )
    with pytest.warns(
        UserWarning,
        match="Expected to be 0 and 1. Values have been Adjusted to 0 and 1 Automaticly.",
    ):
        CEMMatcher(dataframe, ["a"])


def test_non_unique_index() -> None:
    """
    Test CEM with a non-unique index.
    """
    dataframe = pd.DataFrame(
        {"a": [1, 2, 3], "user_id": [1, 1, 3], "treatment": [1, 1, 0]}
    )
    with pytest.raises(ValueError, match="Must Have Unique Values"):
        CEMMatcher(dataframe, ["a"])


if __name__ == "__main__":
    pytest.main()
