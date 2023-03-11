import numpy as np
import statsmodels.discrete.discrete_model as sm
from typing import Union
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
from .utils import model_score

class RFSC_base:
    """
    Base class for RFSC. Used to update RFSC parameters for DRFSC model
    """
    def __init__(
            self, 
            n_models: int=300, 
            n_iters: int=150, 
            tuning: float=50, 
            tol: float=0.002, 
            alpha: float=0.99, 
            rip_cutoff: float=1, 
            metric: str='roc_auc', 
            verbose: bool=False,
        ):
        """
        Parameters
        ----------
        n_models : int 
            Number of models generated per iteration. Default=300.
        n_iters : int 
            Number of iterations. Default is 150.
        tuning : float 
            Learning rate that dictates the speed of regressor inclusion probability (rip) convergence. Smaller values -> slower convergence. Default is 50.
        tol : float 
            Tolerance condition. Default is 0.002.
        alpha : float 
            Significance level for model pruning. Default is 0.99.
        rip_cutoff : float 
            Determines rip threshold for feature inclusion in final model. Default=1.
        metric : str
            Optimization metric. Default='roc_auc'. Options: 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'.
        verbose : bool 
            Provides extra information. Defaults is False.
        """
        
        self.n_models = n_models
        self.n_iters = n_iters
        self.tuning = tuning
        self.tol = tol
        self.alpha = alpha
        self.rip_cutoff = rip_cutoff
        self.metric = metric
        self.verbose = verbose

        if self.metric not in ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']:
            raise ValueError(f"metric must be one of 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'. Received: {self.metric} ")
        if not isinstance(self.n_models, int):
            raise TypeError(f"n_models parameter must be an integer. Received: {type(self.n_models)}")
        if not isinstance(self.n_iters, int):
            raise TypeError(f"n_iters parameter must be an integer. Received: {type(self.n_iters)}")
        if self.tol < 0:
            raise ValueError(f"tol parameter must be a positive number. Received: {self.tol}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha parameter must be between 0 and 1. Received: {self.alpha}") 
        if self.tuning < 0:
            raise ValueError(f"tuning parameter must be a positive number. Received: {self.tuning}")
        if not 0 < self.rip_cutoff <= 1:
            raise ValueError(f"rip_cutoff parameter must be between 0 and 1. Received: {self.rip_cutoff}")
        
        print(
            f"{self.__class__.__name__} Initialised with with parameters: \n \
            n_models = {n_models}, \n \
            n_iters = {n_iters}, \n \
            tuning = {tuning}, \n \
            metric = {metric}, \n \
            alpha = {alpha} \n ------------"
        ) if self.verbose else None
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_models={self.n_models}, n_iters={self.n_iters}, tuning={self.tuning}, metric={self.metric}, alpha={self.alpha})"
    

class RFSC(RFSC_base):
    """
    Implements RFSC algorithm based on parameters inherited from RFSC_base. 
    """
    
    def __init__(self, *args):
        """
        Inherits parameters from RFSC_base. See help(RFSC_base.__init__) for more information.
        """
        super().__init__(*args)

    def set_attr(self, params: dict):
        """
        Setter for RFSC attributes. Used to update RFSC parameters for DRFSC model.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to update.
        """
        for key, value in params.items():
            self.__setattr__(key, value)
        
    def fit_drfsc(self, drfsc_index: tuple):
        if hasattr(self, 'X_train'):
            self.rfsc_main(
                X_train=self.X_train, 
                X_val=self.X_val, 
                Y_train=self.Y_train, 
                Y_val=self.Y_val, 
                drfsc_index=drfsc_index
            )
            self._cleanup()
            return self
        else: 
            raise AttributeError("No data loaded")
            
    def load_data_rfsc(
            self, 
            X_train: np.ndarray, 
            X_val: np.ndarray, 
            Y_train: np.ndarray, 
            Y_val: np.ndarray, 
            feature_partition: list=None, 
            sample_partition: list=None, 
            drfsc_index: tuple=None,
            mu_init: dict=None,
            M: dict=None,
        ) -> None:
        """
        Loads the data passed from DRFSC into the RFSC object.

        Parameters
        ----------
        X_train : np.ndarray
            Training set data
        X_val : np.ndarray
            Validation set data
        Y_train : np.ndarray
            Training set labels
        Y_val : np.ndarray
            Validation set labels
        feature_partition : list 
            list of features indices for corresponding drfsc index
        sample_partition : list
            list of sample indices for corresponding drfsc index
        drfsc_index : tuple
            index of DRFSC bin of the form (r,i,j), where r is the drfsc iteration number, i is the vertical partition index, and j is the horizontal partition index
        mu_init : dict
            dictionary containing user-assigned RIPs. If None, RIPs are initialised to 1/n_features.
        M : dict
            dictionary containing relevant previous information for feature sharing
        """

        if drfsc_index:
            self.drfsc_index = drfsc_index
            feature_share = join_features(
                                        features=feature_partition, 
                                        M=M[drfsc_index[2]]
                                    )
            self.features_passed = [int(i) for i in feature_share]
            self.X_train = X_train[:, self.features_passed][sample_partition, :]
            self.X_val = X_val[:, self.features_passed]
            self.Y_train = Y_train[sample_partition]
            self.Y_val = Y_val
            
            _, n_features  = np.shape(self.X_train)
            self.mu_0 = (1/n_features) * np.ones((n_features))
            if mu_init is not None:
                for idx, feature in enumerate(self.features_passed):
                    if feature in mu_init.keys():
                        self.mu_0[idx] = mu_init[feature]

        else:
            self.X_train = X_train
            self.X_val = X_val
            self.Y_train = Y_train
            self.Y_val = Y_val
            
            _, n_features  = np.shape(self.X_train)
            self.mu_0 = (1/n_features) * np.ones((n_features))

    
    def rfsc_main(
            self, 
            X_train: np.ndarray, 
            X_val: np.ndarray,
            Y_train: np.ndarray,
            Y_val: np.ndarray,
            drfsc_index: tuple=None
        ) -> None:
        """
        This is the main section of the RFSC algorithm called by DRFSC. It extracts the model populations and evaluates them on the validation set, and updates the feature inclusion probabilities accordingly.

        Parameters
        ----------
        X_train : np.ndarray
            Training set data
        X_val : np.ndarray
            Validation set data
        Y_train : np.ndarray
            Training set labels
        Y_val : np.ndarray
            Validation set labels
        drfsc_index : tuple, optional 
            index of DRFSC bin of the form (r,i,j), where r is the drfsc iteration number, i is the vertical partition index, and j is the horizontal partition index

        """
        # initialization
        self.perf_check = 0
        self.rnd_feats = {}
        self.sig_feats = {}        
        avg_model_size = np.empty((0,))
        avg_performance = np.empty((0,))
        mu = self.mu_0
        
        for t in range(self.n_iters):
            mask, performance_vector, size_vector = self.generate_models(
                                                            X_train=X_train,
                                                            Y_train=Y_train,
                                                            X_val=X_val, 
                                                            Y_val=Y_val, 
                                                            mu=mu
                                                        )
            
            mu_update = self.update_feature_probability(
                            mask=mask, 
                            performance=performance_vector, 
                            mu=mu
                        )
            
            avg_model_size = np.append(avg_model_size, np.mean(size_vector.ravel()[np.flatnonzero(performance_vector)]))
            avg_performance = np.append(avg_performance, np.mean(performance_vector.ravel()[np.flatnonzero(performance_vector)]))
            
            if perf_check(t, avg_performance, self.tol):
                self.perf_check += 1
            else:
                self.perf_check = 0
                
            if drfsc_index is None:
                print(f"iter: {t}, avg model size: {avg_model_size[t]:.2f}, avg perf is: {avg_performance[t]:.5f}, tol not reached, max diff is: {np.abs(mu_update - mu).max():.5f}, perf check: {self.perf_check}.") if self.verbose else None
            else:
                print(f"iter: {t} index: {drfsc_index}, avg model size: {avg_model_size[t]:.2f}, avg perf: {avg_performance[t]:.5f}, tol not reached, max diff: {np.abs(mu_update - mu).max():.5f}, perf check: {self.perf_check}.") if self.verbose else None


            if tol_check(mu_update, mu, self.tol): # stop if tolerance is reached.
                print(f"Tol reached. Number of features above rip_cutoff is {np.count_nonzero(mu_update>=self.rip_cutoff)}")
                break   
            
            elif self.perf_check >= 2:
                perf_break = True
                break
            
            mu = mu_update
            
        self.iters = t
        if perf_break is True:
            sig_sorted = {k: v for k, v in sorted(self.sig_feats.items(), key=lambda item: item[1], reverse=True)}
            self.features_ = list(sig_sorted.keys())[0]
        else:
            self.features_ = select_model(mu=mu, rip_cutoff=self.rip_cutoff)

        self.model = sm.Logit(
                        Y_train, 
                        X_train[:, self.features_]
                    ).fit(disp=False, method='lbfgs')
        
        self.coef_ = self.model.params
        return self
    
    def generate_models(
            self, 
            X_train: np.ndarray, 
            Y_train: np.ndarray, 
            X_val: np.ndarray, 
            Y_val: np.ndarray, 
            mu: np.ndarray
        ):
        """
        Generates random models and for each model evaluates the significance of each feature. Statistically significant features are retained and resultant model is regressed again and its performance on validation partition is evaluated and stored.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        Y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation data
        Y_val : np.ndarray
            Validation labels
        mu : np.ndarray
            Array of regressor inclusion probabilities of each feature.
        
        Returns
        -------
        mask_mtx : np.ndarray
            Matrix containing 1 in row i at column j if feature j was included in model i, else 0.
        performance_vector : np.ndarray
            Array containing performance of each model.
        size_vector : np.ndarray
            Array containing number of features in each model.
        """
        
        # initialise vectors
        mask = np.empty((0,))
        mask_mtx = np.zeros((len(mu),)) # mask matrix
        performance_vector = np.zeros((self.n_models,))# performance vector
        size_vector = np.zeros((self.n_models,)) # average model size vector
        mu[0] = 1 # set bias term to 1
        for i in range(self.n_models):
            count = 0
            mask_vector = np.zeros((len(mu),))
            while True:
                generated_features = generate_model(mu) # generate model
                
                
                if tuple(generated_features) not in self.rnd_feats.keys(): # check if model has been generated before
                    logreg_init = sm.Logit(
                                        Y_train, 
                                        X_train[:, generated_features]
                                    ).fit(disp=False, method='lbfgs')
                    significant_features = prune_model(
                                                model=logreg_init, 
                                                feature_ids=generated_features, 
                                                alpha=self.alpha
                                            )
                    self.rnd_feats[tuple(generated_features)] = significant_features
                    
                else: # if model has been generated before, use the stored significant features
                    significant_features = self.rnd_feats[tuple(generated_features)]
                
                if len(significant_features) <= 1:
                    count += 1
                    if count > 1000:
                        self.alpha -= 0.05
                        continue
                else:
                    break

            size_vector[i] = len(significant_features)
            mask_vector[significant_features] = 1
            mask = np.concatenate((mask, mask_vector), axis=0)
            
            if tuple(significant_features) not in self.sig_feats.keys(): # check if model has been evaluated before
                logreg_update = sm.Logit(
                                    Y_train, 
                                    X_train[:, significant_features]
                                ).fit(disp=False, method='lbfgs')
                prediction = logreg_update.predict(X_val[:, significant_features])
                
                performance_vector[i] = model_score(
                                            method=self.metric, 
                                            y_true=Y_val, 
                                            y_pred_label=prediction.round(), 
                                            y_pred_prob=prediction
                                        )
                self.sig_feats[tuple(significant_features)] = performance_vector[i]            
            
            else: # if model has been evaluated before, used the stored performance
                performance_vector[i] = self.sig_feats[tuple(significant_features)]
                    
        mask_mtx = np.reshape(mask, (len(performance_vector), len(mu)))
        return mask_mtx, performance_vector, size_vector

    def update_feature_probability(
        self, 
        mask: np.ndarray, 
        performance: np.ndarray, 
        mu: np.ndarray
    ) -> np.ndarray:
        """
        Updates the feature probability vector mu based on the performance of the models generated.

        Parameters
        ----------
        mask : np.ndarray
            Matrix of shape (n_models, n_features) containing the mask of the models generated.
        performance : np.ndarray
            Performance evaluation for each model.
        mu : np.ndarray
            Current feature probability vector.

        Returns
        -------
        mu_update : np.ndarray 
            Updated feature probability vector.
        """
        features_incld = np.sum(mask, axis=0) #(n_features,)
        features_excld = (np.ones(len(mu)) * self.n_models) - features_incld #(n_features,)
        features_performance = performance @ mask #(n_features,)
        
        ## evaluate importance of features
        with np.errstate(divide='ignore', invalid='ignore'):
            E_J_incld = features_performance / features_incld
            E_J_excld = (sum(performance) - features_performance) / features_excld
        
        # for where features not chosen in any models
        E_J_incld[np.isnan(E_J_incld)] = 0
        E_J_excld[np.isnan(E_J_excld)] = 0
        E_J_excld[np.isinf(E_J_excld)] = 0
        
        gamma = gamma_update(performance=performance, tuning=self.tuning)
        _mu = mu + gamma*(E_J_incld - E_J_excld)
        mu_update = np.asarray([min(max(prob, 0),1) for prob in _mu])
        return mu_update
    
    def predict_proba(self, X_test):
        """
        Predict {0,1} probability of test observations given fitted model.
        """
        return self.model.predict(X_test[:, self.features_])    
    
    def predict_label(self, X_test):
        """
        Predict {0,1} labels of test observations given fitted model.
        """
        return self.model.predict(X_test[:, self.features_]).round()
    
    def _cleanup(self):
        """
        Removes data from RFSC to save memory
        """
        for attr in ['X_train', 'X_val', 'Y_train', 'Y_val', 'rnd_feats', 'sig_feats']:
            if hasattr(self, attr): delattr(self, attr)
            
def perf_check(iter: int, avg_perf: np.ndarray, tol: float) -> bool:
    """
    Checks if performance has converged based on tolerance threshold.

    Parameters
    ----------
    iter : int
        current iteration
    avg_perf : np.ndarray
        average performance vector
    tol : float
        tolerance condition

    Returns
    -------
    (bool): 
        True if performance converged, else False.
    """
    if iter > 2 and np.abs(avg_perf[iter] - avg_perf[iter-1]) <= tol:
        return True
    else:
        return False

            
def tol_check(mu_update: np.ndarray, mu: np.ndarray, tol: float):
    """
    Checks if maximum difference between mu vectors is below tolerance threshold.

    Parameters
    ----------
    mu_update : np.ndarray
        mu at iteration t+1
    mu : np.ndarray
        mu at iteration t
    tol : float
        tolerance condition

    Returns
    -------
    (bool): 
        True max difference below tolerance, else False.
    """
    return np.abs(mu_update - mu).max() < tol
                
def select_model(mu: np.ndarray, rip_cutoff: float) -> list:
    """
    Selects final model based on features that are above the regressor inclusion probability (rip) threshold
    
    Parameters
    ----------
    mu : np.ndarray
        current feature probability vector
    rip_cutoff : float
        regressor inclusion probability threshold
    
    Returns
    -------
    model_feats : list
        list of features that are above the rip threshold
    """
    model_feats = list((mu>=rip_cutoff).nonzero()[0])
    return model_feats

def gamma_update(
        performance: np.ndarray, 
        tuning: float=10
    ) -> float:
    """ 
    Scale the update of the feature probability vector.
    
    Parameters
    ----------
    performance : np.ndarray
        performance evaluation for each model.
    tuning : float, optional
        tuning parameter to adjust convergence rate, default=10
    
    Returns
    -------
    gamma : float
        Scaling factor for the update of the feature probability vector
    """
    gamma = 1/(tuning*(np.max(performance) - np.mean(performance)) + 0.1)
    return gamma

def generate_model(mu: np.ndarray) -> np.ndarray:
    """
    Takes a vector of probabilities and returns a random model.
    
    Parameters
    ----------
    mu : np.ndarray 
        array of probabilities for each feature
    
    Returns
    -------
    index : np.ndarray
        randomly generated numbers corresponding to features ids based on probabilities.
    """
    if np.count_nonzero(mu) == 0:
        raise ValueError("mu cannot be all zeros")
    
    index= [0]
    while len(index) <= 1:
        index = np.flatnonzero(np.random.binomial(1,mu))
    return index

def prune_model(
        model: object, 
        feature_ids: list, 
        alpha: float
    ) -> list:
    """ 
    Tests whether features are signifincant at selected signifincance level. Returns index of significant features.
    
    Parameters
    ----------
    model : object
        logistic regression model object. See statsmodels.api.Logit
    feature_ids : list
        feature ids included in the model.
    alpha : float
        (0,1) significance level.
    
    Returns
    -------
    sig_feature_ids : list
        list of features above the significance level.
    """
    sig_feature_ids = list(set(feature_ids[np.where(model.pvalues<=alpha)]))
    return sig_feature_ids

def join_features(features: list, M: Union[set, int]) -> list:
    """
    Joins the feature partitions to the relevant information from previous iterations.
    """
    if isinstance(M, int):
        return list(set(features).union([M]))
    
    if isinstance(M, set):
        return list(set(features).union(M))
    
 