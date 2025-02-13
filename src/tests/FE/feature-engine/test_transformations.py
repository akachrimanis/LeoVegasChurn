import pandas as pd
from src.FE.feature_engine.transformations.log_transformation import (
    apply_log_transformation,
)


def test_apply_log_transformation():
    data = pd.DataFrame({"A": [1, 10, 100], "B": [1, 2, 3]})
    transformed = apply_log_transformation(data, variables=["A"])
    assert transformed["A"].equals(pd.Series([0.0, 2.302585, 4.605170], name="A"))
