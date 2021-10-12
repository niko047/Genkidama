from sklearn.ensemble import RandomForestClassifier


ALGORITHM_SWITCHER = {
    'random_forest' : RandomForestClassifier,
}

def algorithm_switch(algorithm):
    return ALGORITHM_SWITCHER.get(algorithm, None)
