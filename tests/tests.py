import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from drfsc import drfsc, rfsc, utils
import pytest
from os import path
import pickle
import itertools


@pytest.fixture
def data_import():
    data = utils.load_wdbc("tests/wdbc.data")
    data = pd.DataFrame(data)
    X = data.loc[:, 2:]
    X.columns = [f"x_{i}" for i in range(1, X.shape[1] + 1)]
    Y = data.loc[:, 1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)
    return X_train, X_val, Y_train, Y_val, X_test, Y_test

@pytest.fixture
def DRFSC_creation():
    model = drfsc.DRFSC(n_vbins=2, n_hbins=2, output='single', metric='auprc')
    return model

@pytest.fixture
def data_loading(DRFSC_creation, data_import):
    X_train, X_val, Y_train, Y_val, X_test, Y_test = DRFSC_creation.load_data(*data_import, polynomial=2)
    return DRFSC_creation, X_train, X_val, Y_train, Y_val, X_test, Y_test


@pytest.fixture
def model_fit_single(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    if path.exists("tests/test_model_single.pkl"):
        # if want to retrain model, delete test_model_single.pkl
        model = pickle.load(open("tests/test_model_single.pkl", "rb"))
    else:
        model.fit(X_train, X_val, Y_train, Y_val)
        pickle.dump(model, open("tests/test_model_single.pkl", "wb"))
    return model

@pytest.fixture
def model_fit_ensemble(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    model.output = 'ensemble'
    if path.exists("tests/test_model_ensemble.pkl"):
        # if want to retrain model, delete test_model_ensemble.pkl
        model = pickle.load(open("tests/test_model_ensemble.pkl", "rb"))
    else:
        model.fit(X_train, X_val, Y_train, Y_val)
        pickle.dump(model, open("tests/test_model_ensemble.pkl", "wb"))
    return model

def test_DRFSC_creation(DRFSC_creation):
    assert DRFSC_creation.n_vbins == 2
    assert DRFSC_creation.n_hbins == 2
    assert DRFSC_creation.output == 'single'
    assert DRFSC_creation.metric == 'auprc'
    
def test_check_params():
    with pytest.raises(ValueError):
        model = drfsc.DRFSC(output='test')
        model.set_initial_mu(1.1)
        
def test_data(data_import):
    X_train, X_val, Y_train, Y_val, X_test, Y_test = data_import
    assert X_train.shape[0] == 364
    assert X_val.shape[0] == 91
    assert X_test.shape[0] == 114
    assert Y_train.shape[0] == 364
    assert Y_val.shape[0] == 91
    assert Y_test.shape[0] == 114
    
def test_DRFSC_data_load(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    assert model.loaded_data == 1

def test_distributions(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    dist_features, dist_samples = utils.create_balanced_distributions(
        labels=Y_train, 
        n_feats=X_train.shape[1], 
        n_vbins=model.n_vbins, 
        n_hbins=model.n_hbins
    )
    
    # check that feature distribution is correct
    assert type(dist_features) == np.ndarray
    assert dist_features.shape[1] == model.n_vbins
    assert dist_features.shape[0] >= X_train.shape[1]/model.n_vbins
    
    # check that feature distribution contains all features
    flat_features = dist_features.ravel()
    assert min(flat_features) == 0
    assert max(flat_features) == X_train.shape[1] - 1 # -1 because of 0 indexing
    assert all(i in flat_features for i in range(X_train.shape[1]))
    
    # check that sample distribution is correct
    assert type(dist_samples) == np.ndarray
    assert dist_samples.shape[1] == model.n_hbins
    assert dist_samples.shape[0] >= Y_train.shape[0]/model.n_hbins
    
    # check that sample distribution contains all samples
    flat_samples = dist_samples.ravel()
    assert min(flat_samples) == 0
    assert max(flat_samples) == X_train.shape[0] - 1 # -1 because of 0 indexing
    assert all(i in flat_samples for i in range(Y_train.shape[0]))
    

def test_RFSC_indexing(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    r=1
    hbins_list = list(range(model.n_hbins))
    process_dict = model.generate_processes(r=r, v_bins=model.n_vbins, h_bins=hbins_list)
    assert len(process_dict) == model.n_vbins * model.n_hbins * r
    for i,j in itertools.product(range(model.n_vbins), hbins_list):
        assert (r,i,j) in process_dict
        
def test_model_fit_single(model_fit_single, data_loading):
    *_, X_test, Y_test = data_loading
    assert hasattr(model_fit_single, 'model')
    score = model_fit_single.score(X_test, Y_test)
    assert isinstance(score, dict)
    assert score['metric'] == model_fit_single.metric
    assert hasattr(model_fit_single, 'features_num')
    
def test_model_fit_ensemble(model_fit_ensemble, data_loading):
    *_, X_test, Y_test = data_loading
    score = model_fit_ensemble.score(X_test, Y_test)
    assert isinstance(model_fit_ensemble.ensemble_pred, pd.DataFrame)
    assert model_fit_ensemble.ensemble_pred.shape[0] == len(Y_test)
    assert model_fit_ensemble.ensemble_pred.shape[1] == model_fit_ensemble.n_hbins + 2 # 1 column for each n_hbin + 'mean_prob' + 'majority' columns 
    assert score['metric'] == model_fit_ensemble.metric
    #assert hasattr(model_fit_ensemble, 'features_num')
    
def test_set_initial_mu_float(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    model.set_initial_mu(0.1)
    assert all(value == 0.1 for value in model.mu_init.values())

def test_set_initial_mu_dict_index(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    mu_init = {1:0.1, 2:0.2, 3:0.3}
    model.set_initial_mu(mu_init, _type='index')
    assert len(model.mu_init) == len(mu_init)
    
def test_set_initial_mu_dict_feature(data_loading):
    model, X_train, X_val, Y_train, Y_val, X_test, Y_test = data_loading
    mu_init = {'x_1':0.1, 'x_2':0.2, 'x_3':0.3}
    model.set_initial_mu(mu_init, _type='name')
    assert len(model.mu_init) == len(mu_init)
    
    

    
    