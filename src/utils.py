import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# helper functions for DRFSC
def create_balanced_distributions(
        labels, 
        n_feats: int,
        n_vbins: int, 
        n_hbins: int
    ):
    return vertical_distribution(
                n_feats=n_feats,
                n_vbins=n_vbins
            ), balanced_horizontal_partition(
                labels=labels, 
                n_hbins=n_hbins
            )

def balanced_horizontal_partition(labels, n_hbins: int):
    index_by_label = {
        l: np.where(labels == l)[0] for l in np.unique(labels)
    }
    
    index_by_label_split = {
        key: horizontal_distribution(n_samples=len(value), n_hbins=n_hbins) for key,value in index_by_label.items()
    }

    _partitions = []
    for i in range(n_hbins):
        comined_sample = np.concatenate(([index_by_label_split[key][:, i] for key in index_by_label_split.keys()]), axis=0)
        _partitions.append(random.sample(list(comined_sample), len(comined_sample)))
    
    return np.transpose(np.array(_partitions))

def horizontal_distribution(n_samples: int, n_hbins: int):
    """
    Creates horizontal bins for the dataset.
    
    Parameters
    ----------
        n_samples (int): number of samples in the data.
        n_hbins (int): number of horizontal partiions of the data.
        
    Returns
    -------
        horizontal_partitions (np.ndarray): Contains in each column the sample indexes of the samples that belong to that horizontal bin.
    """
    sample_index = np.arange(n_samples).tolist()# list of feature ids
    rnd_list = random.sample(sample_index, len(sample_index)) # random shuffle of feature ids
    
    _dups = (n_hbins - (len(rnd_list) % n_hbins)) if (n_samples % n_hbins != 0) else 0 # number of duplicates to add

    _comb = rnd_list + random.sample(sample_index, _dups)
    horizontal_partitions = np.reshape(_comb, (int(len(_comb) / n_hbins), n_hbins)) # convert reshuffled features into matrix of dim [int(len(_comb) / n_hbins)x n_bins]
    return horizontal_partitions

def vertical_distribution(n_feats: int, n_vbins: int):
    """
    Function that creates vertical bins for the features in the dataset for use by DRFSC.
    
    Parameters
    ----------
        n_feats (int): number of features in the data
        n_vbins (int): number of vertical partitions of the data.
        
    Returns
    -------
        vertical_partitions (np.ndarray): Contains in each column the features that belong to that vertical bin.
    """

    feature_index = np.arange(1, n_feats).tolist() # list of feature ids
    rnd_list = random.sample(feature_index, len(feature_index)) # random shuffle of feature ids
    
    _dups = (n_vbins - (len(rnd_list) % n_vbins)) if ((n_feats - 1) % n_vbins != 0) else 0 # number of duplicates to add
    
    _comb = rnd_list + random.sample(feature_index, _dups)
    rnd_mat = np.reshape(_comb, (int(len(_comb) / n_vbins), n_vbins)) # convert reshuffled features into matrix of dim [(int(len(_comb) / n_vbins) x n_vbins]
    vertical_partitions = np.vstack((np.zeros((n_vbins,)), rnd_mat))
    return vertical_partitions


def scale_data(data: np.ndarray or pd.DataFrame) -> np.ndarray or pd.DataFrame:
    """ 
    Uses sklearn.preprocessing to rescale feature values to [0,1].
    
    Parameters
    ----------
        data (np.ndarray or pd.DataFrame): data to be transformed
        
    Returns
    -------
        data_out (np.ndarray or pd.DataFrame): [0,1] transform of data   
    """
    minmax = MinMaxScaler(feature_range=(0,1))
    if isinstance(data, pd.DataFrame):
        column_names = minmax.fit(data).get_feature_names_out()
        data_out = pd.DataFrame(minmax.fit_transform(np.asarray(data)), columns=column_names)
    
    else:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be np.ndarray or pd.DataFrame")
        data_out = minmax.fit_transform(data)

    return data_out

def extend_features(data: np.ndarray or pd.DataFrame, degree: int=1) -> np.ndarray or pd.DataFrame:
    """ 
    Uses sklearn.preprocessing to create polynomial features of data and add bias term.
    
    Parameters
    ----------
        data (np.ndarray or pd.DataFrame): data to be transformed
        degree (int, optional): Degree of non-linearity to generate. Defaults to 1. By default just adds a bias term.
        
    Returns
    -------
        data_out (np.ndarray or pd.DataFrame): Polynomial transformed data   
    """

    if not isinstance(degree, int):
        raise TypeError("degree parameter must be an integer")
    
    poly = PolynomialFeatures(degree=degree, interaction_only=False, 
                            include_bias=True, order="C")
    
    if isinstance(data, pd.DataFrame):
        column_names = poly.fit(data).get_feature_names_out()
        data_out = pd.DataFrame(poly.fit_transform(np.asarray(data)), columns=column_names)
    
    else:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be np.ndarray or pd.DataFrame")
        data_out = poly.fit_transform(data)
        
    return data_out
        
def remove_feature_duplication(list_of_arrays: list) -> set:
    return set(np.concatenate(list_of_arrays)) if list_of_arrays else set(list_of_arrays)


def data_info(X_train=None, X_val=None, X_test=None, Y_train=None, Y_val=None, Y_test=None):
    print("Information for Loaded Data: \n -------------")
    print("'X_train' SHAPE: {}".format(X_train.shape)) if X_train is not None else None
    print("          TYPE:  {}".format(type(X_train).__name__)) if X_train is not None else None
    print("'X_val'   SHAPE: {}".format(X_val.shape)) if X_val is not None else None
    print("          TYPE:  {}".format(type(X_val).__name__)) if X_val is not None else None
    print("'X_test'  SHAPE: {}".format(X_test.shape)) if X_test is not None else None
    print("          TYPE:  {}".format(type(X_test).__name__)) if X_test is not None else None
    
    print("'Y_train' SHAPE: {}".format(Y_train.shape, )) if Y_train is not None else None
    print("          TYPE:  {}".format(type(Y_train).__name__)) if Y_train is not None else None
    print("'Y_val'   SHAPE: {}".format(Y_val.shape)) if Y_val is not None else None
    print("          TYPE:  {}".format(type(Y_val).__name__)) if Y_val is not None else None
    print("'Y_test'  SHAPE: {}".format(Y_test.shape)) if Y_test is not None else None
    print("          TYPE:  {} \n -------------".format(type(Y_test).__name__)) if Y_test is not None else None


