# utils/data_split.py
from sklearn.model_selection import train_test_split

def get_target_shadow_split(X, y, seed=42):
    X_temp, X_test_MIA, y_temp, y_test_MIA = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_target, X_shadow, y_target, y_shadow = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    return X_target, X_shadow, y_target, y_shadow, X_test_MIA, y_test_MIA


def get_train_test_split(X_target, y_target, seed=42):
    return train_test_split(X_target, y_target, test_size=0.2, random_state=seed)


def get_retain_forget_split(X_target_train_set, y_target_train_set, seed=42,
                             forget_class=None):
    """forget_class: set to e.g. 7 for CIFAR-10's 'forget class 7' scenario,
       or None to keep the full 20% forget split (Purchase/Texas/MuFac)."""
    X_retain, X_forget, y_retain, y_forget = train_test_split(
        X_target_train_set, y_target_train_set, test_size=0.2,
        random_state=seed, stratify=y_target_train_set)

    if forget_class is not None:
        y_forget = y_forget[y_forget['label'] == forget_class]
        X_forget = X_forget.loc[y_forget.index]

    return X_retain, X_forget, y_retain, y_forget