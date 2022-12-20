import numpy as np
import multiprocessing as mp
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as sm
from src.utils import remove_feature_duplication, scale_data, extend_features, data_info, create_balanced_distributions
from src.rfsc import RFSC, RFSC_base, evaluate_model, model_score, evaluate_interim_model


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
            max_processes=None
        ):
        
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
        self.max_processes = max_processes if max_processes is not None else mp.cpu_count()
        self.RFSC_model = RFSC_base(metric = self.metric)
        
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
        return self.RFSC_model.__dict__
        
    def set_rfsc_params(self, params: dict):
        for key, value in params.items():
            self.RFSC_model.__setattr__(key, value)
        print("Updated RFSC model parameters: {}".format(self.RFSC_model.__dict__))
        
    def generate_processes(self, r: int, v_bins: int, h_bins: list) -> dict:
        return {(r,i,j): RFSC() for i in range(v_bins) for j in h_bins}
    
    def load_data(self, 
            X_train, 
            X_val, 
            Y_train, 
            Y_val, 
            X_test=None, 
            Y_test=None, 
            polynomial=1, 
            preprocess=True
        ):
        
        if preprocess is True:
            if not isinstance(polynomial, int):
                raise ValueError("Polynomial must be an integer")
            
            if polynomial < 0:
                raise ValueError("Polynomial must be greater than 0")
                
            X_train, X_val = map(lambda x: extend_features(scale_data(x), degree=polynomial), [X_train, X_val])
            
            if X_test is not None and X_test.shape[1] != X_train.shape[1]:
                X_test = extend_features(scale_data(X_test), degree=polynomial) 
        
        data_info(
            X_train=X_train, 
            X_val=X_val, 
            Y_train=Y_train, 
            Y_val=Y_val
        )
        
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
        
        Y_train = Y_train.flatten() if Y_train.ndim != 1 else Y_train
        Y_val = Y_val.flatten() if Y_val.ndim != 1 else Y_val
        
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
            
        if isinstance(Y_test, pd.DataFrame):
            Y_test = Y_test.to_numpy()
            Y_test = Y_test.flatten() if Y_test.ndim != 1 else Y_test
        
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
        return X_train, X_val, Y_train, Y_val, X_test, Y_test
    
    def fit(self, X_train, X_val, Y_train, Y_val):

        if self.loaded_data != 1:
            X_train, X_val, Y_train, Y_val, _, _= self.load_data(
                                                        X_train=X_train, 
                                                        X_val=X_val, 
                                                        Y_train=Y_train, 
                                                        Y_val=Y_val, 
                                                        polynomial=self.polynomial, 
                                                        preprocess=self.preprocess
                                                    )

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
                                    feature_partition=distributed_features[:,i], 
                                    sample_partition=distributed_samples[:,j], 
                                    drfsc_index=key, 
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
                print("result_obj length is {}. Should be {}".format(len(result_obj), (self.n_vbins * len(non_converged_hbins))))
            
            for result in result_obj:
                result.model, result.evaluation = evaluate_interim_model(
                                                    model_features = result.features_, 
                                                    X_train = X_train, 
                                                    X_val = X_val,
                                                    Y_train = Y_train, 
                                                    Y_val = Y_val,
                                                    metric = self.metric
                                                ) # predict on all sub-processes
                iter_results[result.drfsc_index] = result
                

            self.results_full, single_iter_results = self.update_full_results(
                                                        results_full = self.results_full, 
                                                        iter_results = iter_results
                                                    ) # update full results dict
            
            single_iter_results = self.map_local_feats_to_gt(
                                    iter_results = single_iter_results, 
                                    r = r, 
                                    hbins = non_converged_hbins
                                ) # map local feature indices to global feature indices
            
            comb_sig_feats_gt = [model[0] for model in single_iter_results.values()]
                    
            self.J_best, self.J_star = update_best_models(
                                            J_best = self.J_best, 
                                            J_star = self.J_star, 
                                            single_iter_results = single_iter_results, 
                                            non_converged_hbins = non_converged_hbins, 
                                            metric = self.metric
                                        ) # update the current best results

            non_converged_hbins = self.convergence_check(
                                        r = r,
                                        J_star = self.J_star, 
                                        non_converged_hbins = non_converged_hbins
                                    ) # update converged horizontal partitions
            
            M = self.feature_share(
                    r = r, 
                    results_full = self.results_full, 
                    comb_sig_feats_gt = comb_sig_feats_gt, 
                    non_converged_hbins = non_converged_hbins, 
                    M = M
                ) # update feature list shared with other partitions
            
            print("M: {}".format(M)) if self.verbose else None
            self.M_history.update([(r, M)])
            
            if len(non_converged_hbins) == 0:
                print("All horizontal partitions have converged. Final iter count: {}".format(r+1))
                break
            
        if self.output == 'single':
            self.features_num = select_single_model(J_best = self.J_best)[0]
            self.model, self.model_score = evaluate_interim_model(
                                                model_features = self.features_num, 
                                                X_train = X_train, 
                                                X_val = X_val, 
                                                Y_train = Y_train, 
                                                Y_val = Y_val, 
                                                metric = self.metric
                                            )
        for value in self.results_full.values(): 
            # remove the features_passed from results_full
            value.pop() 
            
        return self
    
    def score(self, X_train, X_val, X_test, Y_train, Y_val, Y_test):
        for x in [X_train, X_val, X_test, Y_train, Y_val, Y_test]:
            if isinstance(x, pd.DataFrame):
                x = x.to_numpy()
            
        return evaluate_model(
                        model_features=self.features_num,
                        X_train=X_train,
                        X_val=X_val,
                        X_test=X_test,
                        Y_train=Y_train,
                        Y_val=Y_val,
                        Y_test=Y_test,
                        metric=self.metric
                    )
    

    def predict(self, X_test, Y_test = None):
        """_summary_

        Args:
            X_test (_type_): _description_
            Y_test (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.output == 'ensemble':
            if Y_test is None:
                raise ValueError("Y_test must be provided for ensemble output")
            
            model_ensemble, self.ensemble_predictions, self.ensemble_labels, self.combined_evaluation = self.generate_ensemble(
                                        X_test = X_test, 
                                        Y_test = Y_test, 
                                        J_best = self.J_best, 
                                        metric = self.metric, 
                                        h_bins = self.n_hbins
                                    )
            self.model_ensemble = self.map_feature_indices_to_names(
                                        output = self.output, 
                                        final_model = model_ensemble
                                    )
            
            return self.ensemble_predictions, self.ensemble_labels, self.combined_evaluation
            
        else: # output == 'single'
            self.coef_ = self.model.params
            self.label_pred = self.model.predict(X_test[:, self.features_num])
            
            if self.input_feature_names:
                self.features_ = self.map_feature_indices_to_names(
                                        output=self.output, 
                                        final_model=self.features_num
                                    )
            else:
                self.features_ = self.features_num
            
            return self.label_pred.round()
    
    
    def predict_proba(self, X_test, Y_test=None):
        """_summary_

        Args:
            X_test (_type_): _description_
            Y_test (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.output == 'ensemble':
            if Y_test is None:
                raise ValueError("Y_test must be provided for ensemble output")

            model_ensemble, self.ensemble_predictions, self.ensemble_labels, self.combined_evaluation = self.generate_ensemble(
                                        X_test=X_test, 
                                        Y_test=Y_test, 
                                        J_best=self.J_best, 
                                        metric=self.metric, 
                                        h_bins=self.n_hbins
                                    )
            self.model_ensemble = self.map_feature_indices_to_names(
                                        output=self.output, 
                                        final_model=model_ensemble
                                    )
            
            return self.ensemble_predictions, self.ensemble_labels, self.combined_evaluation

        
        else: #self.output == 'single'
            self.coef_ = self.model.params
            self.label_pred = self.model.predict(X_test[:, self.features_num])
            if self.input_feature_names is not None:
                self.features_ = self.map_feature_indices_to_names(
                                        output=self.output, 
                                        final_model=self.features_num
                                    )
            else:
                self.features_ = self.features_num
            
            return self.label_pred
        
    def generate_ensemble(
            self, 
            X_test, 
            Y_test, 
            J_best: dict, 
            metric: str, 
            h_bins: int
        ):
        """_summary_

        Args:
            X_test (_type_): _description_
            Y_test (_type_): _description_
            J_best (dict): _description_
            metric (str): _description_
            h_bins (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        if not all(isinstance(x, np.nd.array) for x in [self.data, X_test, self.labels, Y_test]):
            raise ValueError("All inputs must be numpy arrays")

        # get the best model for each horizontal partition   
        ensemble = {}
        ensemble_probabilities = pd.DataFrame()
        ensemble_labels = pd.DataFrame()
        for h_bin in range(h_bins):
            model = sm.Logit(
                    self.labels, 
                    self.data[:, J_best[h_bin][0]]
                ).fit(disp = False, method = 'lbfgs')
            ensemble.update({f"hbin_{str(h_bin)}_model" : [J_best[h_bin][0], model]})
            prob_prediction = model.predict(X_test[:, J_best[h_bin][0]])
            label_prediction = [round(x) for x in prob_prediction]
            
            ensemble_probabilities[f"hbin_{str(h_bin)}_model"] = prob_prediction
            ensemble_labels[f"hbin_{str(h_bin)}_model"] = label_prediction
            
        ensemble_probabilities['mean_prob'] = ensemble_probabilities.mean(axis=1)
        ensemble_probabilities['majority'] = [round(x) for x in ensemble_probabilities['mean_prob']]
        
        ensemble_labels['mean_label'] = ensemble_labels.mean(axis=1)
        ensemble_labels['majority'] = [round(x) for x in ensemble_labels['mean_label']]
        
        combined_evaluation = model_score(
                                method=metric, 
                                y_true=Y_test, 
                                y_pred_label=ensemble_labels['majority'].values.tolist(), 
                                y_pred_prob=ensemble_probabilities['mean_prob'].values.tolist()
                            )
        
        evals = {'metric': metric, 'evaluation': combined_evaluation}
        return ensemble, ensemble_probabilities, ensemble_labels, evals


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
            iter_results (dict) with global feature indices.
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
            r (int): current iteration
            results_full (dict): dictionary containing the results from all iterations.
            comb_sig_feats_gt (list): list of global feature indices from models in the current iteration.
            non_converged_hbins (list): list of horizontal partition indicies that have not converged.
            
        Returns
        -------
            M (dict): dictionary containing the features to be shared with each bin in the subsequent iteration
        """
        if self.feature_sharing == 'latest':
            M = {i: 0 for i in range(self.n_hbins)} # reset M dict if feature sharing is set to latest
        
        for j in non_converged_hbins:
            if self.output == 'ensemble':
                M[j] = remove_feature_duplication([results_full[(r, i, j)][0] for i in range(self.n_vbins)])
            
            else: #self.output == 'single':
                if self.feature_sharing == 'top_k':
                    top_k_model_feats = [sorted(results_full.values(), key = lambda x: x[1], reverse = True)[i][0] for i in range(min(self.k, len(results_full.values())))]
                    M[j] = remove_feature_duplication(top_k_model_feats)
                
                else:
                    M[j] = remove_feature_duplication(comb_sig_feats_gt)

        return M
    
    def final_model(self, n_features: int, model_ensemble: dict):
        if self.output != 'ensemble':
            raise ValueError("Final model only valid for ensemble output")
        
        df = pd.DataFrame(columns = model_ensemble.keys(), index = range(n_features)) 
        for key, value in model_ensemble.items():
            coefs = value[2].params
            feat_index= value[0]
            for val in zip(feat_index, coefs):
                df.loc[val[0], key] = val[1]
        df.fillna(0, inplace = True)
        df['mean'] = df.mean(axis=1)
        self.model_coef = list(df[df['mean'] != 0]['mean'])
        self.model_features = list(df[df['mean']!= 0].index)
        
    def map_feature_indices_to_names(
            self, 
            output: str, 
            final_model: dict
        ):
        """
        Maps the feature indices to the original feature names, if they exist.
        
        Parameters
        ----------
        output (str): output type
        final_model (dict): _description_


        Returns
        -------

        """
        if output == 'ensemble':
            for key in final_model.keys():
                final_model[key] = [final_model[key][0], [np.array(self.input_feature_names)[x] for x in final_model[key][0]], final_model[key][1]]
            return final_model
                
        else: #output == 'single':
            return [np.array(self.input_feature_names)[x] for x in final_model]
                
    def convergence_check(
            self, 
            r: int, 
            J_star: dict, 
            non_converged_hbins: list
        ):
        
        """
        Checks if the tolerance condition has been met for the current iteration.
        
        Parameters
        ----------
            r (int): current interation.
            J_star (dict): dictionary of best models from each horizontal partition.
            non_converged_hbins (list): list of horizontal partitions that have not converged
            
        Returns: 
            (list) indicies of horizontal partition that have not converged
        """
        hbins_converged = []
        for hbin in non_converged_hbins:
            if max(J_star[hbin] if r > 0 else J_star[hbin]) == 1:
                print(f"Iter {r}. The best model in hbin {hbin} cannot be improved further") if self.verbose else None
                hbins_converged.append(hbin)

            elif r >= 2 and J_star[hbin][r] == J_star[hbin][r-1] and J_star[hbin][r] == J_star[hbin][r-2]:
                print(f"Iter {r}. No appreciable improvement over the last 3 iterations in hbin {hbin}") if self.verbose else None
                hbins_converged.append(hbin)
                
        return [bin for bin in non_converged_hbins if bin not in hbins_converged]
        
    def feature_importance(self):
        """
        Creates a bar plot of the features of the model and their contribution to the final prediction.
        """
        plt.figure()
        plt.title("Feature Importance")
        plt.xlabel("Feature Name")
        plt.ylabel("Feature Coefficient")
        
        if self.output == 'ensemble':
            cols = list(self.input_feature_names)
            cols.insert(0, 'model_id')
            coef = pd.DataFrame(0, columns = cols, index = range(0, self.n_hbins+1))
            for id, key in enumerate(self.model_ensemble.keys()):
                coef.loc[id, 'model_id'] = key
                coef.loc[id, self.model_ensemble[key][1]] = self.model_ensemble[key][2].params
            coef.loc['mean'] = coef.mean()
            coef.loc['mean', 'model_id'] = 'mean'
            coef_list = coef.loc['mean'].to_list()
            coef_list.pop(0)
            
            disp_dict = dict(zip(self.input_feature_names, abs(np.array(coef_list))))        
        
        else: #self.output == 'single':
            disp_dict = dict(sorted(zip(self.features_, abs(self.coef_)), key = lambda x: x[1], reverse = True))
            
        plt.bar(*zip(*disp_dict.items()))
        return plt.show()
            
    def pos_neg_prediction(
            self, 
            data_index
        ):
        """
        Creates a plot of the positive and negative parts of the prediction
        """
        plt.figure()
        plt.title("+/- Predictions Plot (change this label)")
        plt.xlabel("+/-")
        
        sample_data, _ = self.data[data_index, self.features_num], self.labels[data_index]

        y_neg, y_pos = [], []
        for idx, parameter_value in enumerate(self.coef_):
            if parameter_value < 0:
                y_neg.append(parameter_value * sample_data[idx])
            else:
                y_pos.append(parameter_value * sample_data[idx])
                
        y_neg_norm = 1/(1+np.exp(-sum((abs(x) for x in y_neg))))
        y_pos_norm = 1/(1+np.exp(sum((abs(x) for x in y_pos))))
        disp_dict = dict(zip(('y+', 'y-'), (y_pos_norm, y_neg_norm)))
        return plt.bar(*zip(*disp_dict.items()))
        
    def single_prediction(self, data_index):
        """
        Creates a plot of the single prediction of the final model
        """
        plt.figure()
        plt.title("Single Prediction Plot")
        plt.xlabel("Prediction")
        plt.ylabel("Feature Name")
        
        sample_data, _ = self.data[data_index, self.features_num], self.labels[data_index]
        fitted_vals = np.multiply(sample_data, self.coef_)
        disp_dict = dict(zip(self.features_, fitted_vals))
        return plt.bar(*zip(*disp_dict.items()))
    
def update_best_models(
        J_best: dict, 
        J_star: dict, 
        single_iter_results: dict, 
        non_converged_hbins: list, 
        metric: str
    ):
    """
    Compares results from the current iteration against current best models. If a model from the current iteration is better, it is saved.
    
    Returns dictionary containing the best models for each horizontal partition.
    """
    for key, model in single_iter_results.items():
        if key[2] in non_converged_hbins and model[1] > J_best[key[2]][1]:
            print("New best model for hbin {} has {} {} -- Model features {}".format(key[2], metric, model[1], model[0]))
            J_best[key[2]] = [model[i] for i in range(3)]
    for j in non_converged_hbins:
        J_star[j].append(J_best[j][1])
        
    return J_best, J_star

def select_single_model(J_best: dict):
    """
    Returns moel with highest performance evaluation
    """
    return sorted(J_best.values(), key = lambda x: x[1], reverse = True)[0]
    
def validate_data(
        X_train, 
        X_val, 
        Y_train, 
        Y_val, 
        X_test=None, 
        Y_test=None
    ):
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
            
def _check_type(
        X_train, 
        X_val, 
        Y_train, 
        Y_val, 
        X_test=None, 
        Y_test=None
    ):
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
        raise ValueError("metric must be one of 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'")
    
    if not isinstance(n_hbins, int):
        raise TypeError("n_hbins must be an integer")
    
    if not isinstance(n_vbins, int):
        raise TypeError("n_vbins must be an integer")
    
    if output not in ['single', 'ensemble']:
        raise ValueError("output must be one of 'single', 'ensemble'")
    
    if feature_sharing not in ['all', 'latest', 'top_k']:
        raise ValueError("feature_sharing must be one of 'all', 'latest', 'top_k'")
    
    if feature_sharing == 'top_k':
        if k is None: 
            raise ValueError("k must be an integer")
        
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        


        
        

        