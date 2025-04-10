from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42
TEST_SIZE = .2
N_SPLIT = 5

def split_data(X, y, n_split=N_SPLIT, test_size=TEST_SIZE, seed=SEED):
    """
    Split data into train and test and return StratifiedKFold instance and train, test data
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)

    return (cv, (X_train, X_test, y_train, y_test))