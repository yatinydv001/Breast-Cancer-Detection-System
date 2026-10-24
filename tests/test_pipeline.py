from src.data import load_data, split_data
from src.models import get_models


def test_data_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_all_models_reach_90_percent():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.90, f"{name} scored too low"
