import numpy as np
import statsmodels.discrete.discrete_model as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve, auc
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
# import random

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
            upweight: float=1, 
        ):
        """
        Parameters
        ----------
            n_models (int, optional): number of models generated per iteration. Defaults to 300.
            n_iters (int, optional): number of iterations. Defaults to 150.
            tuning (float, optional): learning rate that dictates the speed of regressor inclusion probability (rip) convergence. Smaller values -> slower convergence. Defaults to 50.
            tol (float, optional): tolerance condition. Defaults to 0.002.
            alpha (float, optional): significance level for model pruning. Defaults to 0.99.
            rip_cutoff (float, optional): determines rip threshold for feature inclusion in final model. Defaults to 1.
            metric (str, optional): optimization metric. Defaults to 'roc_auc'.
            verbose (bool, optional): Full description. Defaults to False.
            upweight (float, optional): Upweights initial feature rips. Defaults to 1.
        """
        
        self.n_models = n_models
        self.n_iters = n_iters
        self.tuning = tuning
        self.tol = tol
        self.alpha = alpha
        self.rip_cutoff = rip_cutoff
        self.metric = metric
        self.verbose = verbose
        self.upweight= upweight
        
        # self.max_features = max_features
        # self.enforce_max = False
        
        if self.metric not in ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']:
            raise ValueError("metric must be one of 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'")
        
        if not isinstance(self.n_models, int):
            raise TypeError("n_models parameter must be an integer")

        if not isinstance(self.n_iters, int):
            raise TypeError("n_iters parameter must be an integer")
        
        if self.tol < 0:
            raise ValueError("tol parameter must be a positive float")
        
        if not 0 < self.alpha < 1:
            raise ValueError("alpha parameter must be between 0 and 1") 
        
        if self.tuning < 0:
            raise ValueError("tuning parameter must be a positive float")
        
        if not 0 < self.rip_cutoff <= 1:
            raise ValueError("rip_cutoff parameter must be between 0 and 1")
        
        print(
            f"{self.__class__.__name__} Initialised with with parameters: \n \
            n_models {n_models}, \n \
            n_iters {n_iters}, \n \
            tuning {tuning}, \n \
            metric {metric}, \n \
            alpha {alpha} \n ------------") if self.verbose else None
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_models={self.n_models}, n_iters={self.n_iters}, tuning={self.tuning}, metric={self.metric}, alpha={self.alpha})"
    

class RFSC(RFSC_base):
    """
    Implements RFSC algorithm based on parameters inherited from RFSC_base.
    """
    
    def __init__(self, *args):
        super().__init__(*args)

    def set_attr(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)
        
    def fit_drfsc(self, drfsc_index):
        if hasattr(self, 'X_train'):
            self.rfsc_main(
                X_train=self.X_train, 
                X_val=self.X_val, 
                Y_train=self.Y_train, 
                Y_val=self.Y_val, 
                drfsc_index=drfsc_index
            )
            self._cleanup()
            
        else:
            raise AttributeError("No data loaded")
            
        return self
        
    def load_data_rfsc(
            self, 
            X_train: np.ndarray, 
            X_val: np.ndarray, 
            Y_train: np.ndarray, 
            Y_val: np.ndarray, 
            feature_partition: list=None, 
            sample_partition: list=None, 
            drfsc_index: tuple=None, 
            M=None
        ) -> None:
        
        if drfsc_index:
            self.drfsc_index = drfsc_index
            self.features_passed = join_features(
                                        features=feature_partition, 
                                        M=M[drfsc_index[2]]
                                    )
            self.X_train = X_train[:, self.features_passed][sample_partition, :]
            self.X_val = X_val[:, self.features_passed]
            self.Y_train = Y_train[sample_partition]
            self.Y_val = Y_val

        else:
            self.X_train = X_train
            self.X_val = X_val
            self.Y_train = Y_train
            self.Y_val = Y_val
    
    def rfsc_main(
            self, 
            X_train: np.ndarray, 
            X_val: np.ndarray,
            Y_train: np.ndarray,
            Y_val: np.ndarray,
            drfsc_index=None
        ) -> None:

        self.rnd_feats = {}
        self.sig_feats = {}
        
        
        avg_model_size = np.empty((0,))
        avg_performance = np.empty((0,))
        _, n_features  = np.shape(X_train)
        mu = (1/n_features) * np.ones((n_features))
        mu = mu * self.upweight
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
            
            avg_performance = np.append(avg_performance, np.mean(performance_vector.ravel()[np.flatnonzero(performance_vector)]))
            avg_model_size = np.append(avg_model_size, np.mean(size_vector.ravel()[np.flatnonzero(performance_vector)]))

            if drfsc_index:
                print(f"iter: {t}, avg model size: {avg_model_size[t]:.2f}, tol not reached, avg perf is: {avg_performance[t]:.3f} max diff is: {np.abs(mu_update - mu).max():.5f}") if self.verbose else None
            else:
                print(f"iter: {t} index: {drfsc_index}, avg model size: {avg_model_size[t]:.2f}, tol not reached, avg perf is: {avg_performance[t]:.3f} max diff is: {max(np.abs(mu_update - mu)):.5f},") if self.verbose else None

            if tol_check(mu_update, mu, self.tol): # stop if tolerance is reached.
                print("Tol reached. Number of features above rip_cutoff is {}".format(np.count_nonzero(mu_update >= self.rip_cutoff)))
                break   
            
            if np.mean(performance_vector.ravel()[np.flatnonzero(performance_vector)]) > 0.99: # stop if the average performance is greater than 0.99.
                break
            
            mu = mu_update
            
        self.iters = t
        self.features_ = select_model(
                            mu = mu, 
                            rip_cutoff = self.rip_cutoff
                        )
        
        self.model = sm.Logit(
                        Y_train, 
                        X_train[:, self.features_]
                    ).fit(disp=False, method = 'lbfgs')
        
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
            X_train (np.ndarray) : Training data.
            Y_train (np.ndarray) : Training labels
            X_val (np.ndarray) : Validation data
            Y_val (np.ndarray) : Validation labels.
            mu (np.ndarray): array of regressor inclusion probabilities of each feature
            
        Returns
        -------
            mask_mtx (np.ndarray): matrix containing 1 in row i at column j if feature j was included in model i, else 0
            performance_vector (np.ndarray): array containing performance of each model
            size_vector (np.ndarray): array containing number of features in each model. 
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
                # if self.enforce_max is True and len(generated_features) > self.max_features:
                #     generated_features = random.sample(generated_features, self.max_features)
                    
                if tuple(generated_features) not in self.rnd_feats.keys(): # check if model has been generated before
                    logreg_init = sm.Logit(
                                        Y_train, 
                                        X_train[:, generated_features]
                                    ).fit(disp=False, method = 'lbfgs')
                    significant_features = prune_model(
                                                model = logreg_init, 
                                                feature_ids = generated_features, 
                                                alpha = self.alpha
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
            mask = np.concatenate((mask, mask_vector), axis = 0)
            
            if tuple(significant_features) not in self.sig_feats.keys(): # check if model has been evaluated before
                logreg_update = sm.Logit(
                                    Y_train, 
                                    X_train[:, significant_features]
                                ).fit(disp=False, method = 'lbfgs')
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
            mask (np.ndarray) : matrix of shape (n_models, n_features) containing the mask of the models generated.
            performance (np.ndarray) : performance evaluation for each model.
            mu (np.ndarray) : current feature probability vector.

        Returns
        ----------
            mu_update (np.ndarray) : updated feature probability vector.
        """
        features_incld = np.sum(mask, axis = 0, dtype = np.int_) #(n_features,)
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
        for attr in ['X_train', 'X_val', 'Y_train', 'Y_val']:
            if hasattr(self, attr): delattr(self, attr)
                
        for attr in ['rnd_feats', 'sig_feats']:
            if hasattr(self, attr): delattr(self, attr)
            
def tol_check(mu_update: np.ndarray, mu: np.ndarray, tol: float):
    """
    Checks if maximum difference between mu vectors is below tolerance threshold.

    Parameters
    ----------
        mu_update (np.ndarray): mu at iteration t+1
        mu (np.ndarray): mu at iteration t
        tol (float): tolerance condition

    Returns
    -------
        (bool): True max difference below tolerance, else False.
    """
    return np.abs(mu_update - mu).max() < tol
                
def select_model(mu: np.ndarray, rip_cutoff: float) -> list:
    """
    Selects final model based on features that are above the regressor inclusion probability (rip) threshold
    """
    return list((mu >= rip_cutoff).nonzero()[0])


def gamma_update(
        performance: np.ndarray, 
        tuning: float=10
    ) -> float:
    """ 
    Scale the update of the feature probability vector.
    
    Parameters
    ----------
        performance (np.ndarray) : performance evaluation for each model.
        tuning (float, optional) : tuning parameter to adjust convergence rate, default = 10.
        
    Returns
    ----------
        gamma (float) : scaling factor for the update of the feature probability vector.
    """
    return 1/(tuning*(np.max(performance) - np.mean(performance)) + 0.1)

def generate_model(mu: np.ndarray) -> np.ndarray:
    """
    Takes a vector of probabilities and returns a random model.
    
    Parameters
    ----------
        mu (np.ndarray): array of probabilities for each feature.
        
    Returns
    ----------
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
        model (object) : logistic regression model object. See statsmodels.api.Logit
        feature_ids (list) : feature ids included in the model.
        alpha (float) : (0,1) significance level.
        
    Returns
    ----------
        significant feature ids (list).
    """
    return list(set(feature_ids[np.where(abs(model.tvalues) >= norm.ppf(alpha))]))


def evaluate_model(
        model_features: list, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray, 
        Y_train: np.ndarray, 
        Y_val: np.ndarray, 
        Y_test: np.ndarray, 
        metric: str
    ):
    """
    Evaluates the performance of a model on the test set.

    Parameters
    ----------
        model_features (list) : Subset of features to be included in the model.
        X_train (np.ndarray) : Training data.
        X_val (np.ndarray) : Validation data
        X_test (np.ndarray) : Test data.
        Y_train (np.ndarray) : Training labels.
        Y_val (np.ndarray) : Validation labels.
        Y_test (np.ndarray) : Test labels.
        metric (str) : {'acc', 'roc_auc', 'avg_prec','f1', 'auprc'}. Metric used to evaluate model performance.

    Returns
    -------
        model_final (object) : Fitted logistic regression model. See statsmodels.Logit.
        model_performance (float) : Performance of the model on the validation set.
    """
    
    model_final = sm.Logit(
                        np.concatenate((Y_train, Y_val), axis=0), 
                        np.concatenate((X_train, X_val), axis=0)[:, model_features]
                    ).fit(disp=False, method='lbfgs')
    
    label_prediction = model_final.predict(X_test[:, model_features]) # predict probabilities
    
    return model_final, model_score(
                            method=metric, 
                            y_true=Y_test, 
                            y_pred_label=label_prediction.round(), 
                            y_pred_prob= label_prediction
                        )
    
def evaluate_interim_model(
            model_features: list, 
            X_train: np.ndarray, 
            X_val: np.ndarray, 
            Y_train: np.ndarray, 
            Y_val: np.ndarray, 
            metric: str
        ):
    """
    Evaluates the performance of a model on the validation set.

    Parameters
    ----------
        model_features (list) : Subset of features to be included in the model.
        X_train (np.ndarray) : Training data.
        X_val (np.ndarray) : Validation data
        Y_train (np.ndarray) : Training labels.
        Y_val (np.ndarray) : Validation labels.
        metric (str) : {'acc', 'roc_auc', 'avg_prec','f1', 'auprc'}. Metric used to evaluate model performance.

    Returns
    -------
        model_final (object) : Fitted logistic regression model. See statsmodels.Logit.
        model_performance (float) : Performance of the model on the validation set.
    """
    model_final = sm.Logit(
                    Y_train, 
                    X_train[:, model_features]
                ).fit(disp = False, method = 'lbfgs')

    label_prediction = model_final.predict(X_val[:, model_features]) 
    
    return model_final, model_score(
                            method=metric, 
                            y_true=Y_val, 
                            y_pred_label=label_prediction.round(), 
                            y_pred_prob=label_prediction
                        )

def model_score(
        method: str, 
        y_true: np.ndarray,
        y_pred_label: np.ndarray, 
        y_pred_prob: np.ndarray
    ):
    
    """
    Evalutates model performance based on specified metric using sklearn.metrics.
    
    Parameters
    ----------
        method (str) : {'acc', 'roc_auc', 'avg_prec','f1', 'auprc'}. Metric used to evaluate model performance.
        y_true (np.ndarray) : {0,1} ground truth labels.
        y_pred_label (np.ndarray) : {0,1} predicted labels.
        y_pred_prob (np.ndarray) : [0,1] predicted probabilities.
        
    Returns
    -------
        Value in range [0,1] representing model performance, based on specified metric.
    """
    methods = {
        'acc' : accuracy_score(
                    y_true=y_true, 
                    y_pred=y_pred_label
                ), \
        'roc_auc' : roc_auc_score(
                    y_true=y_true, 
                    y_score=y_pred_prob, 
                    average='weighted'
                ), \
        'avg_prec' : average_precision_score(
                    y_true=y_true, 
                    y_score=y_pred_prob, 
                    average='weighted'
                ), \
        'f1' : f1_score(
                    y_true=y_true, 
                    y_pred=y_pred_label, 
                    average='binary'
                ), \
        'auprc' : au_prc(
                    y_true=y_true, 
                    y_pred_prob=y_pred_prob
                )}
    return methods.get(method, 'Invalid method')

def au_prc(y_true: np.ndarray, y_pred_prob: np.ndarray):
    """
    Computes the area under the precision-recall curve

    Parameters
    ----------
        y_true (np.ndarray): array of {0,1} ground truth labels.
        y_pred_prob (np.ndarray): array of [0,1] predicted probabilities.

    Returns
    -------
        float: area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)
    return auc(x=recall, y=precision)

def join_features(features: list, M) -> list:
    if isinstance(M, int):
        return list(set(features).union([M]))
    
    if isinstance(M, set):
        return list(set(features).union(M))
    
 