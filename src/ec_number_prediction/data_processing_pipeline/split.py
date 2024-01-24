from logging import Logger
import logging
import os
import luigi
import numpy as np
import pandas as pd

from skmultilearn.model_selection import IterativeStratification

from ec_number_prediction.data_processing_pipeline.n_classes_removal import NClassesRemoval

logger = logging.getLogger('luigi-interface')


class StratifiedSplit(luigi.Task):

    def requires(self):
        return NClassesRemoval()
    
    def input(self):
        return luigi.LocalTarget('dataset_binarized_filtered_without_n.csv')

    def output(self):
        return [
                luigi.LocalTarget('splits/train.csv'), 
                luigi.LocalTarget('splits/test.csv'),
                luigi.LocalTarget('splits/validation.csv'),
                luigi.LocalTarget('splits/before_split_dataset_stats_test.html'),
                luigi.LocalTarget('splits/after_split_dataset_stats_test.html'),
                luigi.LocalTarget('splits/before_split_dataset_stats_final.html'),
                luigi.LocalTarget('splits/after_split_dataset_stats_final.html')
                ]
    
    def apply_stratification_sklearn(self, X, y, test_size=0.15, train_size=0.85):
        
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, train_size])
        train_indexes, test_indexes = next(stratifier.split(X, y))
        X_train = X.iloc[train_indexes]
        y_train = y.iloc[train_indexes, :]

        X_test = X.iloc[test_indexes]
        y_test = y.iloc[test_indexes, :]

        return X_train, y_train, X_test, y_test
    
    def correct_split(self, X_train, y_train, X_test, y_test, df_with_stats, validation = False):

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

    def generate_stats(self, y_train, y_test, y_val=None):
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

        os.makedirs("splits", exist_ok=True)

        #################### first split ####################
        logger.info("First split")
        X_train, y_train, X_test, y_test = self.apply_stratification_sklearn(X, y)

        logger.info("First split: generating stats")
        df_with_stats, table_styled = self.generate_stats(y_train, y_test)
        table_styled.to_html("splits/before_split_dataset_stats_test.html")

        #################### first split correction ####################
        logger.info("First split correction")
        X_train, y_train, X_test, y_test = self.correct_split(X_train, y_train, X_test, y_test, df_with_stats)

        logger.info("First split correction: generating stats")
        df_with_stats, table_styled = self.generate_stats(y_train, y_test)
        table_styled.to_html("splits/after_split_dataset_stats_test.html")

        #################### second split ####################
        logger.info("second split")
        X_train, y_train, X_val, y_val = self.apply_stratification_sklearn(X_train, y_train, test_size=0.25, train_size=0.75)

        df_with_stats, table_styled = self.generate_stats(y_train, y_test, y_val)
        table_styled.to_html("splits/before_split_dataset_stats_final.html")

        train = dataset[dataset["accession"].isin(X_train)]
        validation = dataset[dataset["accession"].isin(X_val)]
        test = dataset[dataset["accession"].isin(X_test)]

        #################### second split correction ####################
        logger.info("second split correction")
        X_train, y_train, X_val, y_val = self.correct_split(X_train, y_train, X_val, y_val, df_with_stats, validation=True)

        logger.info("second split correction: generating stats")
        df_with_stats, table_styled = self.generate_stats(y_train, y_test, y_val)
        table_styled.to_html("splits/after_split_dataset_stats_final.html")

        #################### save splits ####################

        logger.info("saving splits")
        train = dataset[dataset["accession"].isin(X_train)]
        validation = dataset[dataset["accession"].isin(X_val)]
        test = dataset[dataset["accession"].isin(X_test)]

        train.to_csv("splits/train.csv", index=False)
        validation.to_csv("splits/validation.csv", index=False)
        test.to_csv("splits/test.csv", index=False)

        

        