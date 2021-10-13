from sklearn.model_selection import GridSearchCV

SEARCH_SWITCHER = {
    'grid_search_cv' : GridSearchCV,
}

def search_switch(search : str):
    return SEARCH_SWITCHER.get(search, None)