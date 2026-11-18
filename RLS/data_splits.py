import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = ROOT
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(ROOT, 'figures')

ROUND_STEP = 0.5                         # recoveries are rounded to the half-star grid
TEST_RATIO, SPLIT_SEED = 0.20, 42        # random 80/20 split of the ratings


def load_movielens_1m(data_dir=DATA_DIR):
    """Returns (train, test): dense user x movie matrices, 0 = missing. Ratings are integers 1..5."""
    path = os.path.join(data_dir, 'ratings.dat')
    if not os.path.exists(path):
        raise FileNotFoundError(f"ratings.dat not found in {data_dir}; download MovieLens-1M from https://grouplens.org/datasets/movielens/1m/")
    df = pd.read_csv(path, sep='::', header=None, engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    n_users, n_movies = int(df.user_id.max()), int(df.movie_id.max())     # 6040 x 3952 (matrix indexed by movie id)

    rng = np.random.default_rng(SPLIT_SEED)
    test_idx = rng.choice(len(df), size=int(TEST_RATIO * len(df)), replace=False)
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True

    def to_matrix(part):
        m = np.zeros((n_users, n_movies))
        m[part.user_id.values - 1, part.movie_id.values - 1] = part.rating.values.astype(float)
        return m

    train, test = to_matrix(df[~is_test]), to_matrix(df[is_test])
    assert not np.any((train > 0) & (test > 0)), "train and test overlap"
    print(f"MovieLens-1M: {n_users} users x {n_movies} movies; train {int((train > 0).sum()):,}, test {int((test > 0).sum()):,}")
    return train, test

