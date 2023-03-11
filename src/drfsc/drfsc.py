import numpy as np
import multiprocessing as mp
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as sm
from typing import Union, Tuple
from .utils import *
from .rfsc import RFSC, RFSC_base

class DRFSC:
    """
        Distributed Randomised Feature Selection for Classification (DRFSC)
    """
    def __init__(
            self, 
            n_vbins: int=1, 
            n_hbins: int=1, 
            n_runs: int=1, 
            redistribute_features: bool=False, 
            feature_sharing: str='all', 
            k: int=0, 
            output: str='single', 
            metric: str='roc_auc', 
            verbose: bool=False, 
            polynomial: int=1, 
            preprocess: bool=True, 
            max_processes: int=None
        ):
        """
        Constructor for DRFSC class. Initialises the DRFSC class with the given parameters.

        Parameters
        ----------
        n_vbins : int, optional
            Number of vertical partitions to create for the data. Defaults to 1.
        n_hbins : int, optional
            Number of horizontal partitions to create for the data. If output = 'ensemble', each hbin will converge to its own best model. Defaults to 1.
        n_runs : int, optional
            Number of feature-sharing iterations to perform. Larger numbers may yield better results, but also take longer. Defaults to 1.
        redistribute_features : bool, optional
            If True, the base features included in each bin will be shuffled at each feature-sharing iteration. Does not affect feature sharing. Defaults to False.
        feature_sharing : str, optional 
            The method used to share features. Defaults to 'all'. Options (str): 'all', 'latest', 'top_k'. If feature_sharing = 'all', the entire history of best features from all sub-processes will be shared. If feature)sharing = 'latest', features from all sub-processes at the current iteration will be shared. If feature_sharing = 'top_k', the best k features will be shared. 
        k : int, optional 
            Number of best features to share. Only used if feature_sharing = 'top_k'. Defaults to 0.
        output : str, optional
            Output type desired. Options (str): 'single', 'ensemble'. If output = 'single', the best model from all sub-processes will be returned. If output = 'ensemble', the best model from each horizontal partition will be returned. If output = 'ensemb;e'. no features between different horizontal partitions will be created. Defaults to 'single'.
        metric : str, optional
            Evaluation metric used in the optimisation process. Options (str) : ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']. Defaults to 'roc_auc'. For more information on the metrics, see the documentation for the sklearn.metrics module.
        verbose : bool, optional
            if True, prints extra information. Defaults to False.
        polynomial : int, optional
            degree of polynomial features to use. Defaults to 1.
        preprocess : bool, optional
            If True, will scale-data, create dummies for categorical variables, and create polynomial features based on passed `polynomial` parameter. If False, will only convert data to numpy arrays. Defaults to True.
        max_processes : int, optional
            Enforces maximum number of processes that can be generated. If None, will use all available cores. Defaults to None.
        """
        
        validate_model(
            metric=metric, 
            n_hbins=n_hbins, 
            n_vbins=n_vbins, 
            output=output, 
            feature_sharing=feature_sharing,
            k=k
        )
            
        self.metric = metric
        self.n_vbins = n_vbins
        self.n_hbins = n_hbins
        self.n_runs = n_runs
        self.redistribute_features = redistribute_features
        self.feature_sharing = feature_sharing
        self.k = k
        self.output = output
        self.verbose = verbose
        self.polynomial = polynomial
        self.preprocess = preprocess
        self.loaded_data = 0
        self.max_processes = max(max_processes, mp.cpu_count()) if max_processes is not None else mp.cpu_count()
        self.RFSC_model = RFSC_base(metric = self.metric, verbose=self.verbose)
        
        print(f"{self.__class__.__name__} Initialised with parameters: \n \
            n_vbins = {n_vbins}, \n \
            n_hbins = {n_hbins}, \n \
            n_runs = {n_runs}, \n \
            redistribute = {redistribute_features}, \n \
            sharing = {feature_sharing}, \n \
            k = {k}, \n \
            output = {output}, \n \
            metric = {metric}, \n \
            max_processes is {self.max_processes} \n ------------") if self.verbose else None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_vbins={self.n_vbins}, n_hbins={self.n_hbins}, n_runs={self.n_runs}, redistribute_features={self.redistribute_features}, feature_sharing={self.feature_sharing}, k={self.k}, output={self.output}, metric={self.metric}, verbose={self.verbose}, polynomial={self.polynomial}, preprocess={self.preprocess}, max_processes={self.max_processes})"
        
    def get_rfsc_params(self):
        """
        Getter for RFSC parameters. Returns a dictionary of the current RFSC parameters
        """
        return self.RFSC_model.__dict__
        
    def set_rfsc_params(self, params: dict):
        """
        Setter for RFSC parameters. Updates the RFSC parameters with the given dictionary. Dictionary must be in the form of {parameter_name: parameter_value}.
        
        To view available parameters, see help(RFSC_base.__init__)
        """
        for key, value in params.items():
            self.RFSC_model.__setattr__(key, value)
        print(f"Updated RFSC model parameters: {self.RFSC_model.__dict__}")
        
    def set_initial_mu(self, mu_init: Union[dict, float], _type: str='name') -> None:
        """
        Setter for initial mu values for RFSC. 
        
        If ``type(mu_init) == float``, will set all mu values to that float. 
        
        If ``type(mu_init) == dict``, will set the mu values for the given features to the given values. 
        
        If dict is passed, ``_type`` must be set to 'name' or 'index'. If 'name', the keys of the dictionary must be the names of the features. If 'index', the keys of the dictionary must be the indices of the features.

        Parameters
        ----------
        mu_init : Union[dict, float] 
            Initial mu values for RFSC. Can be a dictionary or float.
            
        _type : str, optional
            _Only required if type(mu_init) == dict_. Type of keys in mu_init. 
        """
        if not isinstance(mu_init, (dict, float)):
            raise TypeError(f"mu_init must be a dictionary or float. Received type: {type(mu_init)}.")
        
        if self.input_feature_names is None:
            feat_names = np.arange(self.data.shape[1])
        else:
            feat_names = self.input_feature_names
        
        if isinstance(mu_init, float):
            if mu_init < 0 or mu_init > 1:
                raise ValueError(f"mu_init must be between 0 and 1. Value '{mu_init}' is invalid.")
            self.mu_init = pd.Series(data=mu_init, index=feat_names).to_dict()
            self.mu_init_num = pd.Series(data=mu_init, index=range(len(feat_names))).to_dict()
            return
        
        if _type not in ['name', 'index']:
            raise ValueError(f"Invalid _type '{_type}' for {type(mu_init)}. _type must be 'name' or 'index'.")
        
        self.mu_init = pd.Series(data=0, index=feat_names).to_dict()
        self.mu_init_num = pd.Series(data=0, index=range(len(feat_names))).to_dict()
        
        if any(value < 0 or value > 1 for value in mu_init.values()):
            raise ValueError(f"all initial mu value's must be between 0 and 1.")
        
        if _type == 'name':
            for idx, key in enumerate(self.mu_init.keys()):
                if key in mu_init.keys():
                    if mu_init[key] == 0:
                        continue
                    self.mu_init[key] = mu_init[key]
                    self.mu_init_num[idx] = mu_init[key]

        else:
            for idx, key in enumerate(self.mu_init_num.keys()):
                if key in mu_init.keys():
                    if mu_init[key] == 0:
                        continue
                    self.mu_init_num[key] = mu_init[key]
                    
        self.mu_init_num = {k: v for k, v in self.mu_init_num.items() if v != 0}
        self.mu_init = {feat_names[k]: v for k, v in self.mu_init_num.items()}            

    def generate_processes(self, r: int, v_bins: int, h_bins: list) -> dict:
        return {(r,i,j): RFSC() for i in range(v_bins) for j in h_bins}
    
    def load_data(self, 
            X_train: Union[np.ndarray, pd.DataFrame], 
            X_val: Union[np.ndarray, pd.DataFrame], 
            Y_train: Union[np.ndarray, pd.DataFrame], 
            Y_val: Union[np.ndarray, pd.DataFrame], 
            X_test: Union[np.ndarray, pd.DataFrame]=None, 
            Y_test: Union[np.ndarray, pd.DataFrame]=None, 
            polynomial:int=1, 
            preprocess:bool=True
        ):
        """
        Preprocesses the data in the required way for the DRFSC model. Can be used to load data into the model if it has not been loaded yet. Scales the data to [0,1] and creates polynomial expansion based on the passed 'polynomial' parameter. 

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame 
            Train set data
        X_val : np.ndarray or pd.DataFrame
            Validation set data
        Y_train : np.ndarray or pd.DataFrame
            Train set labels
        Y_val : np.ndarray or pd.DataFrame
            Validation set labels
        X_test : np.ndarray or pd.DataFrame, optional
            Test set data. Only required if postprocessing is required. Defaults to None.
        Y_test : np.ndarray or pd.DataFrame optional
            Test set labels. Only required if postprocessing is required. Defaults to None.
        polynomial : int, optional
            degree of polynomial features to use. Defaults to 1.
        preprocess : bool, optional
            If True, will scale-data, create dummies for categorical variables, and create polynomial features based on passed `polynomial` parameter. If False, will only convert data to numpy arrays. Defaults to True.

        Returns 
        -------
        X_train, X_val, Y_train, Y_val, X_test, Y_test : np.ndarray
            Preprocessed data and labels
        """
        
        if preprocess is True:
            if not isinstance(polynomial, int):
                raise ValueError("Polynomial must be an integer")
            
            if polynomial < 0:
                raise ValueError("Polynomial must be greater than 0")
                
            X_train, X_val = map(lambda x: extend_features(scale_data(x), degree=polynomial), [X_train, X_val])
            
            if X_test is not None and X_test.shape[1] != X_train.shape[1]:
                X_test = extend_features(scale_data(X_test), degree=polynomial) 
    
        self.input_feature_names = None
        self.input_label_names = None
        
        if any(isinstance(x, pd.DataFrame) for x in [X_train, X_val, Y_train, Y_val]):
            _check_type(
                X_train=X_train, 
                X_val=X_val, 
                X_test=X_test, 
                Y_train=Y_train, 
                Y_val=Y_val, 
                Y_test=Y_test
            )
            
            self.input_feature_names = X_train.columns.to_numpy()
                              
            if isinstance(Y_train, pd.DataFrame):
                if Y_train.shape[1] != Y_train.select_dtypes(include=np.number).shape[1]:
                    Y_train = pd.get_dummies(Y_train.select_dtypes(exlude=["float", 'int']))

                self.input_label_names = Y_train.columns.to_numpy()
                
            elif isinstance(Y_train, pd.Series): # set input_label_names for pd.Series
                self.input_label_names = Y_train.name 
                
            X_train, X_val, Y_train, Y_val = map(lambda x: x.to_numpy(), [X_train, X_val, Y_train, Y_val])

        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
            
        if isinstance(Y_test, (pd.DataFrame, pd.Series)):
            Y_test = Y_test.to_numpy()
            
        Y_test = Y_test.flatten() if Y_test.ndim != 1 else Y_test
        
        self.mu_init = None
        self.mu_init_num = None
        
        validate_data(
            X_train=X_train, 
            X_val=X_val, 
            X_test=X_test, 
            Y_train=Y_train, 
            Y_val=Y_val, 
            Y_test=Y_test
        )
        
        self.loaded_data = 1
        self.labels = np.concatenate((Y_train, Y_val), axis=0)
        self.data = np.concatenate((X_train, X_val), axis=0)
        
        data_info(
            X_train=X_train, 
            X_val=X_val, 
            Y_train=Y_train, 
            Y_val=Y_val
        )
        return X_train, X_val, Y_train, Y_val, X_test, Y_test
    
    def view_data(self):
        if self.loaded_data != 1:
            raise ValueError("Data has not been loaded yet. Run DRFSC.load_data() first.")
        return pd.DataFrame(self.data, columns=self.input_feature_names if self.input_feature_names is not None else np.arange(self.data.shape[1]))
    
    def fit(
            self, 
            X_train: Union[np.ndarray, pd.DataFrame], 
            X_val: Union[np.ndarray, pd.DataFrame],  
            Y_train: Union[np.ndarray, pd.DataFrame], 
            Y_val: Union[np.ndarray, pd.DataFrame]
        ):
        """
        The main function for fitting the model. Returns the a single final model if output == 'single', else returns a model ensemble based on the number of horizontal partitions (n_hbins).

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame 
            Train set data
        X_val : np.ndarray or pd.DataFrame
            Validation set data
        Y_train : np.ndarray or pd.DataFrame
            Train set labels
        Y_val : np.ndarray or pd.DataFrame
            Validation set labels

        """
        if self.loaded_data != 1:
            X_train, X_val, Y_train, Y_val, _, _= self.load_data(
                                                        X_train=X_train, 
                                                        X_val=X_val, 
                                                        Y_train=Y_train, 
                                                        Y_val=Y_val, 
                                                        polynomial=self.polynomial, 
                                                        preprocess=self.preprocess
                                                    )

        if any(isinstance(x, pd.DataFrame) for x in [X_train, X_val, Y_train, Y_val]):
            raise TypeError("Data must be type np.ndarray")
        
        n_samples, n_features = X_train.shape
        
        # create vertical and horizontal paritions
        distributed_features, distributed_samples = create_balanced_distributions(
                                                        labels=Y_train, 
                                                        n_feats=n_features, 
                                                        n_vbins=self.n_vbins, 
                                                        n_hbins=self.n_hbins
                                                    ) 
        
        # Initializations
        self.J_star = {i: [] for i in range(self.n_hbins)}
        self.J_best = {i: [0,0] for i in range(self.n_hbins)} #
        self.results_full = {}
        self.M_history = {}
        M = {i: 0 for i in range(self.n_hbins)} 
        non_converged_hbins = np.arange(self.n_hbins).tolist()
        
        if self.verbose:
            print(f"Number of Samples: {n_samples}. Horizontal Disitribution SHAPE: {np.shape(distributed_samples)}")
            print(f"Number of Features: {n_features}. Vertical Distribution SHAPE: {np.shape(distributed_features)}")
            
            
        if self.n_hbins == 1 and self.n_vbins == 1:
            if self.output == 'ensemble':
                print("WARNING: Ensemble output is not possible with n_hbins = 1 and n_vbins = 1. Setting output = 'single'")
            self.output = 'single'

            result = RFSC()
            result.set_attr(self.RFSC_model.__dict__)
            result.load_data_rfsc(X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val)
            result.rfsc_main(X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val)
            result.model, result.evaluation = evaluate_interim_model(result.features_, X_train, X_val, Y_train, Y_val, self.metric)
            self.J_best = {0: [result.features_, result.evaluation, result.metric]}
            self.build_final_model(J_best=self.J_best)
            return self
            
            
        for r in range(self.n_runs):
            iter_results = {} # initialise dictionary for storing results
            
            subprocesses = self.generate_processes(
                                    r=r, 
                                    v_bins=self.n_vbins, 
                                    h_bins=non_converged_hbins
                                )
            
            if self.redistribute_features:
                distributed_features, _= create_balanced_distributions(
                                            labels=Y_train, 
                                            n_feats=n_features, 
                                            n_vbins=self.n_vbins, 
                                            n_hbins=self.n_hbins
                                        )
                
            # loads data to all sub-processes
            for key in subprocesses.keys():  
                _,i,j = key
                subprocesses[key].set_attr(self.RFSC_model.__dict__)
                subprocesses[key].load_data_rfsc(
                                    X_train=X_train, 
                                    X_val=X_val, 
                                    Y_train=Y_train, 
                                    Y_val=Y_val, 
                                    feature_partition=list(distributed_features[:,i]), 
                                    sample_partition=list(distributed_samples[:,j]), 
                                    drfsc_index=key,
                                    mu_init=self.mu_init_num,
                                    M=M
                                )
            result_obj = [] 
            def store_results(obj): #callback for mp
                result_obj.append(obj)
                
            pool = mp.Pool(processes=min((self.n_vbins * len(non_converged_hbins)), self.max_processes), maxtasksperchild=1)
            
            for i,j in itertools.product(range(self.n_vbins), non_converged_hbins):
                pool.apply_async(
                        RFSC.fit_drfsc, 
                        args=(subprocesses[(r,i,j)], (r,i,j)), 
                        callback=store_results
                    )
            pool.close() # close the pool to new tasks
            pool.join()
        
            if len(result_obj) != (self.n_vbins * len(non_converged_hbins)):
                print(f"result_obj length is {len(result_obj)}. Should be {(self.n_vbins * len(non_converged_hbins))}")           
                                
            for result in result_obj:
                # predict on all sub-processes
                result.model, result.evaluation = evaluate_interim_model(
                                                    model_features=result.features_, 
                                                    X_train=X_train, 
                                                    X_val=X_val,
                                                    Y_train=Y_train, 
                                                    Y_val=Y_val,
                                                    metric=self.metric
                                                ) 
                iter_results[result.drfsc_index] = result

                    
            # update full results dict
            self.results_full, single_iter_results = self.update_full_results(
                                                        results_full=self.results_full, 
                                                        iter_results=iter_results
                                                    ) 
            for i,j in itertools.product(range(self.n_vbins), non_converged_hbins):
                if (r,i,j) not in [x.drfsc_index for x in result_obj]:
                    print(f"missing result {(r,i,j)}")
                    iter_results.update({(r,i,j): [[0], 0,  self.output, [0]]})

            # map local feature indices to global feature indices
            single_iter_results = self.map_local_feats_to_gt(
                                    iter_results=single_iter_results, 
                                    r=r, 
                                    hbins=non_converged_hbins
                                ) 
            
            comb_sig_feats_gt = [model[0] for model in single_iter_results.values()]
            
            # update the current best results
            self.J_best, self.J_star = update_best_models(
                                            J_best=self.J_best, 
                                            J_star=self.J_star, 
                                            single_iter_results=single_iter_results, 
                                            non_converged_hbins=non_converged_hbins, 
                                            metric=self.metric
                                        ) 
            
            # update converged horizontal partitions
            non_converged_hbins = self.convergence_check(
                                        r=r,
                                        J_star=self.J_star, 
                                        non_converged_hbins=non_converged_hbins
                                    ) 
            
            # update feature list shared with other partitions
            M = self.feature_share(
                    r=r, 
                    results_full=self.results_full, 
                    comb_sig_feats_gt=comb_sig_feats_gt, 
                    non_converged_hbins=non_converged_hbins, 
                    M=M
                ) 
            
            print(f"M: {M}") if self.verbose else None
            self.M_history.update([(r, M)])
            
            if len(non_converged_hbins) == 0:
                print(f"All horizontal partitions have converged. Final iter count: {r+1}")
                break
            
        self.build_final_model(J_best=self.J_best)
            
        for value in self.results_full.values(): 
            # remove the features_passed from results_full
            value.pop() 
            
        return self
    
    def build_final_model(self, J_best):
        """
        Builds the final model based on the output specified.

        Parameters
        ----------
        output : str
            Output type. Options: 'single', 'ensemble'
        J_best : dict
            Dictionary containing the best model for each horizontal partition.
        """
        
        if self.output == 'ensemble':
            ensemble = {}
            for h_bin in range(self.n_hbins):
                model = sm.Logit(
                        self.labels, 
                        self.data[:, J_best[h_bin][0]]
                    ).fit(disp=False, method='lbfgs')
                ensemble.update({f"model_h{str(h_bin)}" : [J_best[h_bin][0], model]})
                
            self.ensemble = self.map_feature_indices_to_names(output=self.output, final_model=ensemble)
            self.final_model(self.ensemble)
            self.features_num = self.model_features_num

        else:
            self.features_num = select_single_model(J_best=J_best)[0]
            
        if self.input_feature_names is not None:
            self.features_ = self.map_feature_indices_to_names(
                                    output='single', 
                                    final_model=self.features_num
                                )
        else:
            self.features_ = self.features_num
        #self.features_ = self.map_feature_indices_to_names(output='single', final_model=self.features_num)
        self.model = sm.Logit(
            self.labels, 
            self.data[:, self.features_num]
        ).fit(disp=False, method='lbfgs')
        self.coef_ = self.model.params

    def score(
        self,
        X_test: Union[np.ndarray, pd.DataFrame], 
        Y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
        metric: str=None
    ) -> dict:
        """
        Used to evaluate the final model on the test set.

        Parameters
        ----------
        X_test : np.ndarray or pd.DataFrame
            Test set data
        Y_test : np.ndarray or pd.DataFrame or pd.Series
            Test set labels
        metric : str, optional
            Metric to use for evaluation. By default uses the metric specified in the constructor. Other options: ('acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc').

        Returns
        -------
        evaluation : dict
            returns the score of the model based on the metric specified.
        """
        
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
            
        if isinstance(Y_test, (pd.DataFrame, pd.Series)):
            Y_test = Y_test.to_numpy()
        Y_test = Y_test.flatten() if Y_test.ndim != 1 else Y_test

        eval_metric = metric if metric is not None else self.metric
        score = model_score(
                    method=eval_metric, 
                    y_true=Y_test, 
                    y_pred_label=self.predict(X_test), 
                    y_pred_prob=self.predict_proba(X_test)
                )   
            
        return {'metric': eval_metric, 'score': score}
                
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Uses the best model to predict on the test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test set data.

        Returns
        -------
        ret : np.ndarray
            The predicted labels.
        """
        if self.output == 'ensemble':
            self.ensemble_pred = self.generate_ensemble(X_test=X_test, )
            ret = self.ensemble_pred['majority'].to_numpy()
            
        else: # output == 'single'
            self.label_pred = self.predict_proba(X_test).round()
            ret = self.label_pred
        return ret
    
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Uses the best model to predict on the test set, returns labels.

        Parameters
        ----------
        X_test : np.ndarray
            Test set data.
        
        Returns
        -------
        proba_ : np.ndarray
            The predicted probabilities.
        """
        if self.output == 'ensemble':
            self.ensemble_pred = self.generate_ensemble(X_test=X_test)
            proba_ = self.ensemble_pred['mean_prob'].to_numpy()
            
        else: #self.output == 'single'
            self.label_prob = self.model.predict(X_test[:, self.features_num])
            proba_ = self.label_prob
            
        return proba_
        
    def generate_ensemble(self, X_test: np.ndarray):
        """
        Generates the model ensemble based on the best model for each horizontal partition.

        Parameters
        ----------
        X_test : np.ndarray 
            Test set data

        Returns
        -------
        ensemble : dict
            dictionary that contains as keys the names of the models making up the ensemble. For each key, the value is a list containing a list of the feature indices used for that model, and the model object itself.
        ensemble_proba : pd.DataFrame
            dataframe containing the predicted probabilities for each model in the ensemble
        """
        
        if not all(isinstance(x, np.ndarray) for x in [self.data, X_test, self.labels]):
            raise TypeError("All inputs must have type np.ndarray")

        if not hasattr(self, 'ensemble'):
            raise AttributeError("No ensemble has been generated yet. Please run the .fit method first.")
        # get the best model for each horizontal partition   
        ensemble_proba = pd.DataFrame()
        for k, v in self.ensemble.items():
            features, _, model = v
            ensemble_proba[k] = model.predict(X_test[:, features])
            
        ensemble_proba['mean_prob'] = ensemble_proba.mean(axis=1)
        ensemble_proba['majority'] = [round(x) for x in ensemble_proba['mean_prob']]
    
        return ensemble_proba

    def update_full_results(
            self, 
            results_full: dict, 
            iter_results: dict
        ):
        """
        Updates the full results dictionary with the results from the current iteration
        """
        single_iter_results = {
            result.drfsc_index : 
                [result.features_, result.evaluation, result.metric, result.features_passed] for result in iter_results.values()
        }
        results_full = results_full | single_iter_results
        return results_full, single_iter_results
    
    def map_local_feats_to_gt(
            self, 
            iter_results: dict, 
            r: int, 
            hbins: list
        ) -> dict:
        """
        Maps local feature indices to global feature indices for each model in the current iteration.
        
        Returns
        -------
            iter_results : dict 
                dict updated with global feature indices.
        """
        for i,j in itertools.product(range(self.n_vbins), hbins):
            iter_results[(r,i,j)][0] = list(np.array(iter_results[(r,i,j)][3])[list(iter_results[(r,i,j)][0])])
    
        return iter_results
            
    
    def feature_share(
            self, 
            r: int, 
            results_full: dict, 
            comb_sig_feats_gt: list, 
            non_converged_hbins: list, 
            M: dict
        ):
        """ 
        Computes the features to be shared with each bin in the subsequent iteration.
        
        Parameters
        ----------
        r : int
            current iteration
        results_full : dict
            dictionary containing the results from all iterations
        comb_sig_feats_gt : list
            list of global feature indices from models in the current iteration
        non_converged_hbins : list 
            list of horizontal partition indicies that have not converged
            
        Returns
        -------
        M : dict 
            dictionary containing the features to be shared with each bin in the subsequent iteration
        """
        if self.feature_sharing == 'latest':
            M = {i: 0 for i in range(self.n_hbins)} # reset M dict if feature sharing is set to latest
        
        for j in non_converged_hbins:
            if self.output == 'ensemble':
                M[j] = remove_feature_duplication([results_full[(r, i, j)][0] for i in range(self.n_vbins)])
            
            else: #self.output == 'single':
                if self.feature_sharing == 'top_k':
                    top_k_model_feats = [sorted(results_full.values(), key=lambda x: x[1], reverse=True)[i][0] for i in range(min(self.k, len(results_full.values())))]
                    M[j] = remove_feature_duplication(top_k_model_feats)
                
                else:
                    M[j] = remove_feature_duplication(comb_sig_feats_gt)

        return M
    
    def final_model(self, model_ensemble: dict) -> None:
        """
        Helper function for generating the final model based on the ensemble of models.

        Parameters
        ----------
        model_ensemble : dict
            contains the ensemble of models. Each key is a separate model, and the value is a list containing a list of the feature indices used for that model, the mapped feature names, and the model object itself.
        """

        if self.output != 'ensemble':
            raise ValueError("Final model only valid for ensemble output")
        
        if self.input_feature_names is not None:
            idx = self.input_feature_names
        else:
            idx = range(self.data.shape[1])
    
        df = pd.DataFrame(columns=model_ensemble.keys(), index=idx) 
        df2 = pd.DataFrame(columns=model_ensemble.keys(), index=idx) 
        for key, value in model_ensemble.items():
            coefs = value[2].params
            feat_index = value[0]
            feat_names = value[1]
            for val in zip(feat_index, coefs):
                df.loc[val[0], key] = val[1]
            for val in zip(feat_names, coefs):
                df2.loc[val[0], key] = val[1]
        df.fillna(0, inplace=True)
        df2.fillna(0, inplace=True)
        df['mean'] = df.mean(axis=1)
        df2['mean'] = df2.mean(axis=1)
        
        self.model_coef = np.array(df[df['mean'] != 0]['mean'])
        self.model_features_num = list(df[df['mean']!= 0].index)
        self.model_features_name = list(df2[df2['mean']!= 0].index)
        
    def map_feature_indices_to_names(
            self, 
            output: str, 
            final_model: Union[dict, list]
        ):
        """
        Maps the feature indices to the original feature names, if they exist.
        
        Parameters
        ----------
        output :str
            output type. Either 'single' or 'ensemble'.
        final_model : dict or list
            contains the converged models from the final iteration for each horizontal parition. If output is 'single', it is a single list, containing a list of feature indices and model object. If output is 'ensemble', it is a dictionary, with the keys being the horizontal partition index, and the values being a list containing the feature indices, and model object

        Returns
        -------
        final_model : dict or list
            returns final_model with the feature indices mapped to the original feature names.
        """
        if output == 'ensemble':
            for key in final_model.keys():
                final_model[key] = [final_model[key][0], [np.array(self.input_feature_names)[x] for x in final_model[key][0]], final_model[key][1]]
                
        else: #output == 'single':
            final_model = [np.array(self.input_feature_names)[x] for x in final_model]
            
        return final_model
                
    def convergence_check(
            self, 
            r: int, 
            J_star: dict, 
            non_converged_hbins: list
        ) -> list:
        """
        Checks if the tolerance condition has been met for the current iteration.
        
        Parameters
        ----------
        r : int
            current interation number
        J_star : dict
            dictionary of best models from each horizontal partition.
        non_converged_hbins : list
            list of horizontal partitions that have not converged
            
        Returns
        -------
        hbins_not_converged : list  
            indicies of horizontal partition that have not converged
        """
        hbins_converged = []
        for hbin in non_converged_hbins:
            if max(J_star[hbin] if r > 0 else J_star[hbin]) == 1:
                print(f"Iter {r}. The best model in hbin {hbin} cannot be improved further") if self.verbose else None
                hbins_converged.append(hbin)

            elif r >= 2 and J_star[hbin][r] == J_star[hbin][r-1] and J_star[hbin][r] == J_star[hbin][r-2]:
                print(f"Iter {r}. No appreciable improvement over the last 3 iterations in hbin {hbin}") if self.verbose else None
                hbins_converged.append(hbin)
        
        hbins_not_converged = [bin for bin in non_converged_hbins if bin not in hbins_converged]
        return hbins_not_converged 
        
    def feature_importance(self) -> plt.figure:
        """
        Creates a bar plot of the features of the model and their contribution to the final prediction.
        
        Returns
        Returns
        -------
        figure : matplotlib figure
            hisogram of feature coefficients for features of the final model.
        """
        plt.figure()
        plt.title("Feature Importance")
        plt.xlabel("Feature Name")
        plt.ylabel("Feature Coefficient (abs value)")
        
        try:
            if self.input_feature_names is not None:
                plot_feats = self.features_
            else:
                print("No feature names provided. Plotting feature indices instead")
                plot_feats = self.features_num
            disp = dict(sorted(zip(plot_feats, abs(self.coef_)), key=lambda x: x[1], reverse=True))
            
        except AttributeError:
            raise AttributeError("Model has not been fit yet. Run .fit() first")  

        return plt.bar(*zip(*disp.items()))
            
    def pos_neg_prediction(
            self, 
            data_index: int=0,
            X_test: Union[np.ndarray, pd.DataFrame]=None
        ) -> plt.figure:
        """
        Creates a plot of the positive and negative parts of the prediction. Returned figure shows, for a given sample (indexed by data_index), the value of the coefficients and multiplies them by the feature values. These components of the prediction are then plotted.
        
        Parameters
        ----------
        data_index : int
            Index of the data observation to be plotted. If X_test is not provided, then the index is relative to the provided training/validation data. If X_test is provided, then the index is relative to the provided test data. Default is 0.
        X_test : np.array or pd.DataFrame
            Test data to be used for prediction. If provided, then the index is relative to the provided test data. Default is None.
        
        Returns
        -------
        figure : matplotlib.pyplot.figure
            output figure
        """
    
        if X_test is not None:
            if not isinstance(X_test, (np.ndarray, pd.DataFrame)):
                raise TypeError("X_test must be type np.ndarray or pd.DataFrame")
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.to_numpy()
                
            sample_data =  X_test[data_index, self.features_num]

        else:
            sample_data = self.data[data_index, self.features_num]

        y_neg, y_pos = [], []
        for idx, parameter_value in enumerate(self.coef_):
            if parameter_value < 0:
                y_neg.append(parameter_value * sample_data[idx])
            else:
                y_pos.append(parameter_value * sample_data[idx])
                
        y_neg_norm = 1/(1+np.exp(-sum((abs(x) for x in y_neg))))
        y_pos_norm = 1/(1+np.exp(sum((abs(x) for x in y_pos))))
        disp = dict(zip(('y+', 'y-'), (y_pos_norm, y_neg_norm)))
        
        plt.figure()
        plt.title("+/- Positive and Negative Parts of Prediction")
        figure = plt.bar(*zip(*disp.items()))
        return figure
        
    def single_prediction(            
            self, 
            data_index: int=0,
            X_test: Union[np.ndarray, pd.DataFrame]=None
        ) -> plt.figure:
        """
        Creates a plot of the single prediction of the final model. Figure shows for a given sample (indexed by data_index) the coefficients of the final model, weighted by the feature values for the indexed observation.
        
        Parameters
        ----------
        data_index : int
            Index of the data observation to be plotted. If X_test is not provided, then the index is relative to the provided training/validation data. If X_test is provided, then the index is relative to the provided test data. Default is 0.
        X_test : np.array or pd.DataFrame
            Test data to be used for prediction. If provided, then the index is relative to the provided test data. Default is None.
        
        Returns
        -------
        figure : matplotlib.pyplot.figure
            output figure
        """     
            
        if X_test is not None:
            if not isinstance(X_test, (np.ndarray, pd.DataFrame)):
                raise TypeError("X_test must be type np.ndarray or pd.DataFrame")
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.to_numpy()
        
            sample_data = X_test[data_index, self.features_num]
        else:
            sample_data = self.data[data_index, self.features_num]
        
        try:
            disp = dict(zip(self.features_, np.multiply(sample_data, self.coef_)))
        except AttributeError:
            disp = dict(zip(self.features_num, np.multiply(sample_data, self.coef_)))
            
        plt.figure()
        plt.title(f"Single Prediction Plot for Sample {data_index}")
        plt.ylabel("Sample-weighted prediction")
        plt.xlabel("Feature Name")
        figure = plt.bar(*zip(*disp.items()))
        return figure
    
def update_best_models(
        J_best: dict, 
        J_star: dict, 
        single_iter_results: dict, 
        non_converged_hbins: list, 
        metric: str
    ) -> Tuple[dict, dict]:
    """
    Compares results from the current iteration against current best models. If a model from the current iteration is better, it is saved.
    
    Returns
    -------
    J_best : dict
        dictionary containing as keys the horizontal partition index and as values the best model for that partition (list) in terms of feature indices, performance evaluation (float), and metric used for evaluation (str)
    J_star : dict
        dictionary containing only the performance evaluation of the best model for each horizontal partition (list)
    """
    for key, model in single_iter_results.items():
        if key[2] in non_converged_hbins and model[1] > J_best[key[2]][1]:
            print(f"New best model for hbin {key[2]}. {metric}={round(model[1], 5)} -- Model features {model[0]}")
            J_best[key[2]] = [model[i] for i in range(3)]
    for j in non_converged_hbins:
        J_star[j].append(J_best[j][1])
        
    return J_best, J_star

def select_single_model(J_best: dict) -> list:
    """
    Returns model with highest performance evaluation
    
    Parameters
    ----------
    J_best : dict
        dictionary containing as keys the horizontal partition index and as values the best model for that partition (list) in terms of feature indices, performance evaluation (float), and metric used for evaluation (str)
    
    Returns
    -------
    best_model : list
        List containing the best model for the entire dataset.
    """
    best_model = sorted(J_best.values(), key=lambda x: x[1], reverse=True)[0]
    return best_model
    
def validate_data(X_train, X_val, Y_train, Y_val, X_test=None, Y_test=None):
    """ 
    Checks data is of correct shape
    """
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(f"X_train rows {X_train.shape[0]} must match Y_train {Y_train.shape[0]}")
    if X_val.shape[0] != Y_val.shape[0]:
        raise ValueError("X_val and Y_val must have the same number of rows")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must have the same number of columns")
    if X_test is not None and Y_test is not None:
        if X_test.shape[0] != Y_test.shape[0]:
            raise ValueError("X_test and Y_test must have the same number of rows")
        if X_test.shape[1] != X_train.shape[1]:
            raise ValueError("X_test and X_train must have the same number of columns")
            
def _check_type(X_train, X_val, Y_train, Y_val, X_test=None, Y_test=None):
    """
    Checks data is of same type.
    """
    if type(X_train) != type(X_val):
        raise TypeError(f"types {type(X_train)} and {type(X_val)} do not match")
    if type(Y_train) != type(Y_val):
        raise TypeError(f"types {type(Y_train)} and {type(Y_val)} do not match")
    if X_test is not None and type(X_test) != type(X_train):
        raise TypeError(f"types {type(X_test)} and {type(X_train)} do not match")
    if Y_test is not None and type(Y_test) != type(Y_train):
        raise TypeError(f"types {type(Y_test)} and {type(Y_train)} do not match")
    if Y_test is not None and type(Y_test) != type(Y_val):
        raise TypeError(f"types {type(Y_test)} and {type(Y_val)} do not match")

def validate_model(
        metric: str, 
        n_hbins: int, 
        n_vbins: int, 
        output: str, 
        feature_sharing: str, 
        k: int
    ):
    """
    Checks DRFSC initialised correctly
    """
    if metric not in ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']:
        raise ValueError(f"metric must be one of: ('acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'). Received {metric}")
    if not isinstance(n_hbins, int):
        raise TypeError(f"n_hbins must be an integer. Received {type(n_hbins)}")
    if not isinstance(n_vbins, int):
        raise TypeError(f"n_vbins must be an integer. Received {type(n_vbins)}")
    if output not in ['single', 'ensemble']:
        raise ValueError(f"output must be one of: ('single', 'ensemble'). Received {output}")
    if n_hbins == 1 and output == 'ensemble':
        raise ValueError(f"output must be 'single' if n_hbins is 1.")
    if feature_sharing not in ['all', 'latest', 'top_k']:
        raise ValueError(f"feature_sharing must be one of: ('all', 'latest', 'top_k'). Received {feature_sharing}")
    if feature_sharing == 'top_k':
        if k is None: 
            raise ValueError(f"k must be an integer. Received None")
        if not isinstance(k, int):
            raise TypeError(f"k must be of type int. Received {type(k)}")