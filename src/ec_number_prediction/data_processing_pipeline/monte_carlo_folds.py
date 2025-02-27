from logging import Logger
import logging
import os
from typing import Any, List, Tuple
import luigi
import numpy as np
import pandas as pd

from skmultilearn.model_selection import IterativeStratification

logger = logging.getLogger('luigi-interface')


class MonteCarloFoldSplit(luigi.Task):

    input_file = luigi.Parameter(default="merged_dataset.csv")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def input(self):
        return luigi.LocalTarget(self.input_file)

    def output(self):
        return [
                luigi.LocalTarget('monte_carlo_splits/train.csv'), 
                luigi.LocalTarget('monte_carlo_splits/test.csv'),
                luigi.LocalTarget('monte_carlo_splits/before_split_dataset_stats_final.html'),
                luigi.LocalTarget('monte_carlo_splits/after_split_dataset_stats_final.html')
                ]
    
    def apply_stratification_sklearn(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.15, train_size: float = 0.85, n_splits=2) -> List[Tuple[np.ndarray,
                                                                                                                                     np.ndarray,
                                                                                                                                     np.ndarray,
                                                                                                                                     np.ndarray]]:
        """
        Parameters
        ----------
        X : np.ndarray
            Samples
        y : np.ndarray
            Labels
        test_size : float, optional
            Size of the test set, by default 0.15
        train_size : float, optional
            Size of the train set, by default 0.85
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, y_train, X_test, y_test
        """

        
        stratifier = IterativeStratification(n_splits=n_splits, order=1, sample_distribution_per_fold=[test_size]*n_splits)

        folds = []
        for train_indexes, test_indexes  in stratifier.split(X, y):
            X_train = X.iloc[train_indexes]
            y_train = y.iloc[train_indexes, :]

            X_test = X.iloc[test_indexes]
            y_test = y.iloc[test_indexes, :]

            folds.append((X_train, y_train, X_test, y_test))

        return folds
    
    def correct_split(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                      df_with_stats: pd.DataFrame, validation : bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                                        np.ndarray]:
        """
        Parameters
        ----------
        X_train : np.ndarray
            Samples of the train set
        y_train : np.ndarray
            Labels of the train set
        X_test : np.ndarray
            Samples of the test set
        y_test : np.ndarray
            Labels of the test set
        df_with_stats : pd.DataFrame
            DataFrame with the stats of the split
        validation : bool, optional
            If the split is a validation split, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, y_train, X_test, y_test
        """


        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        X_test_copy = X_test.copy()
        y_test_copy = y_test.copy()

        for i in range(15):
            if validation:
                ecs = df_with_stats[(df_with_stats["Percentage of data"] <= i) & (df_with_stats["Percentage of data"] > i - 1) 
                                & (df_with_stats["variable"] == "Validation relative split")].loc[:,"EC"]
            else:
                ecs = df_with_stats[(df_with_stats["Percentage of data"] <= i) & (df_with_stats["Percentage of data"] > i - 1) 
                                & (df_with_stats["variable"] == "Test relative split")].loc[:,"EC"]
            print(i)
            for ec in ecs:
                cases = y_train_copy[y_train_copy[ec] == 1]
                n_samples = round(((15-i)/100) *cases.shape[0], 3)
                indexes = cases.sample(int(n_samples), random_state=i).index
                X_test_copy = pd.concat((X_test_copy, X_train_copy.loc[indexes]))
                y_test_copy = pd.concat((y_test_copy, y_train_copy.loc[indexes, :]))

                X_train_copy = X_train_copy.drop(indexes)
                y_train_copy = y_train_copy.drop(indexes)

        return X_train_copy, y_train_copy, X_test_copy, y_test_copy

    def generate_stats(self, y_train: np.ndarray, y_test: np.ndarray, y_val: np.ndarray=None) -> Tuple[pd.DataFrame, Any]:
        """
        Parameters
        ----------
        y_train : np.ndarray
            Labels of the train set
        y_test : np.ndarray
            Labels of the test set
        y_val : np.ndarray, optional
            Labels of the validation set, by default None
        
        Returns
        -------
        Tuple[pd.DataFrame, Any]
            DataFrame with the stats of the split, styled table
        """
        y_test_sum = np.sum(y_test)
        y_train_sum = np.sum(y_train)

        sum_of_all = pd.DataFrame([y_train_sum, y_test_sum], index=["train", "test"])

        if y_val is not None:
            y_val_sum = np.sum(y_val)
            sum_of_all = pd.DataFrame([y_train_sum, y_test_sum, y_val_sum], index=["train", "test", "validation"])
            sum_of_all.loc['Validation relative split', :] = sum_of_all.loc['validation', :] / (sum_of_all.loc['train', :] + sum_of_all.loc['test', :] + sum_of_all.loc['validation', :]) * 100
            sum_of_all.loc['Test relative split', :] = sum_of_all.loc['test', :] / (sum_of_all.loc['train', :] + sum_of_all.loc['test', :]+ sum_of_all.loc['validation', :]) * 100
            sum_of_all.loc['Train relative split', :] = sum_of_all.loc['train', :] / (sum_of_all.loc['train', :] + sum_of_all.loc['test', :]+ sum_of_all.loc['validation', :]) * 100

        else:
            sum_of_all.loc['Test relative split', :] = sum_of_all.loc['test', :] / (sum_of_all.loc['train', :] + sum_of_all.loc['test', :]) * 100
            sum_of_all.loc['Train relative split', :] = sum_of_all.loc['train', :] / (sum_of_all.loc['train', :] + sum_of_all.loc['test', :]) * 100

        df = pd.melt(sum_of_all.T.reset_index(), id_vars=['index']).rename(columns={'index': 'EC', 'value': 'Percentage of data'})
        if y_val is not None:
            df = df[(df["variable"]!="train") & (df["variable"]!="validation") & (df["variable"]!="test")]
        else: 
            df = df[(df["variable"]!="train") & (df["variable"]!="test")]

        df1 = sum_of_all.loc['Test relative split', :].describe()
        df2 = sum_of_all.loc['Train relative split', :].describe()
        if y_val is not None:
            df3 = sum_of_all.loc['Validation relative split', :].describe()
            stats_table = pd.concat([df1, df2, df3], axis=1)
        else:
            stats_table = pd.concat([df1, df2], axis=1)

        stats_table.drop(['count'], inplace=True)
        table_styled = stats_table.style.background_gradient(cmap="YlGn")
        

        return df, table_styled



    def run(self):
        
        logger.info("Reading dataset")
        dataset = pd.read_csv(self.input().path)
        X = dataset.loc[:, "accession"]
        y = dataset.iloc[:, 8:]
        y = y.astype(float).astype(int)

        os.makedirs("monte_carlo_splits", exist_ok=True)

        #################### first split ####################
        logger.info("First split")
         
        folds = self.apply_stratification_sklearn(X, y, n_splits=5)

        for i, fold in enumerate(folds):
            X_train, y_train, X_test, y_test = fold

            logger.info("First split: generating stats")
            df_with_stats, table_styled = self.generate_stats(y_train, y_test)
            table_styled.to_html(f"monte_carlo_splits/before_split_dataset_stats_final_{i}.html")

            # #################### first split correction ####################
            # logger.info("First split correction")
            # X_train, y_train, X_test, y_test = self.correct_split(X_train, y_train, X_test, y_test, df_with_stats)

            # logger.info("First split correction: generating stats")
            # df_with_stats, table_styled = self.generate_stats(y_train, y_test)
            # table_styled.to_html("monte_carlo_splits/after_split_dataset_stats_final.html")

            del y_test
            del y_train

            train = dataset[dataset["accession"].isin(X_train)]
            test = dataset[dataset["accession"].isin(X_test)]

            del X_train
            del X_test

            train.to_csv(f"monte_carlo_splits/train_{i}.csv", index=False)
            test.to_csv(f"monte_carlo_splits/test_{i}.csv", index=False)

            del train
            del test

        

        