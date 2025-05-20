"""
Class dedicate to partition the dataset into the desired classes. The input dataset must have all the preprossing.
"""
import pathlib
import numpy as np
from typing import Optional, Union, Tuple, List
import pandas as pd
from iterative_regression.utils import ahi_class
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import json

class TargetSelectionSplitting:
    def __init__(self,
                 dataset: pd.DataFrame,
                 # selection: str,
                 method: str,
                 discrete_column: str,
                 cont_column: str,
                 output_path: pathlib.Path,
                 num_classes: Optional[Union[int, None]] = None,
                 stratify: Optional[Union[str, None]] = None,
                 percent: Optional[Union[float, None]] = None,
                 equal_samples_class: bool = False,
                 ):
        """
        This TargetSelectionSplitting class built for a dataset where the target (AHI) is given as a continuous measure.

        It leverages the discrete definition of the AHI to select the sample of each train, validation and test set.
        The class creates a column of the target as the discrete level, in this scope we use the ahi_class.py file.

        For each given ordinal level of the AHI, depending on the selection parameter, samples from the continuous
        target will base on how we sample the ordinal samples.

        If selection == inter domain, we sample the continuous target where the * are placed

        |-------------------------------------------------------------------| target continuous
                ********                 ********             ********
        |---------------------|-----------------------|----------------------| target ordinal
            level 1             level 2             level 3

         If selection == boundaries

         |-------------------------------------------------------------------| target continuous
          ***            ***********               ********                 *******
        |---------------------|-----------------------|-----------------------| target ordinal


         This helps to select AHI values that in the border threshold or if they in the average area.

         With the method parameter we select if we want to use percentage of the ordinal level to sample given
         the selection method or if we want to use another method. Currently, only percentage is employed.

        :param dataset: pd.Dataframe, dataset to partition
        :param selection:
        :param method:
        :param discrete_column:
        :param cont_column:
        :param percent:
        :param stratify: str, column to stratify the data splits
        :param equal_samples_class: bool, If true then splits will have same number of groups samples across the splits,
        this is done by clipping all the sets to the group with smallest samples.
        """
        self.dataset = dataset
        # self.selection = selection
        self.discrete_column = discrete_column
        self.cont_column = cont_column
        self.method = method
        self.method_percent = percent
        self.stratify = stratify
        self.num_class = num_classes
        self.seed = 42
        self.equal_samples_class = equal_samples_class
        self.output_path = output_path
        self.indexes_splits = {}
        # if not self.selection in ['boundaries', 'interior_domain',
        #                           'oversample_minority', 'undersample_majority', 'smote', 'ADASYN']:
        #     raise ValueError(f'Select an acceptable selection approach')

        # if not self.method in ['percentage']:
        #     raise ValueError(f'Select an acceptable method approach')

        if self.method not in ['boundaries', 'interior_domain',
                               'oversample_minority', 'undersample_majority', 'smote', 'ADASYN', 'all']:
            raise ValueError(f'Select an acceptable selection approach')

        if self.method in ['boundaries', 'interior_domain']:
            if 0 > self.method_percent > 1.0 or isinstance(self.method_percent, int):
                raise ValueError(f'percent must be a float between 0 and 1')
            if percent is not None:
                self.method_percent = percent
            else:
                self.method_percent = 0.40

        if self.dataset[self.discrete_column].nunique() > 10:
            raise ValueError(f'The column {self.discrete_column} is not discrete, '
                             f'Number of unique values found: {self.dataset[self.discrete_column].nunique()}')
        self._sklearn_imbalance_methods = ['oversample_minority', 'undersample_majority', 'smote', 'ADASYN']
        self._boundary_selection_methods = ['boundaries', 'interior_domain']
    def split_train_val_test(self,
                             save_plot: Optional[bool] = True,
                             plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into train, validation, and test sets after selecting subsets based on the discrete
        target column. The function optionally plots the distribution of the continuous columns based on the
        selection criteria applied. It handles imbalances by optionally equalizing class distributions across
        splits.

        The process includes creating a discrete target column if it does not exist, selecting specific subsets
        based on a predefined selection criteria, and then splitting the dataset. If equal_samples_class is set,
        the function balances the class distribution in each split.

        :param plot:
        :return:
        """
        # create the discrete target column if it does not exist
        if self.discrete_column not in self.dataset.columns:
            self.dataset[self.discrete_column] = ahi_class(x=self.dataset[self.discrete_column],
                                                           num_class=self.num_class).values
        else:
            print(f'Discrete column already in dataset')

        if self.method in self._sklearn_imbalance_methods:
            # implement sklearn imbalanced samplers
            train_df, val_df, test_df = self._split_dataset_train_val_test(
                dataset=self.dataset,
                stratify=self.stratify,
                test_size=.20,
                val_size=.10,
                shuffle=True,
                seed=self.seed
            )
            old_train = train_df[[self.discrete_column, self.cont_column]].copy()
            train_df = self._split_and_resample_sklearn_samplers(x=train_df,
                                                                 y=train_df[self.discrete_column],
                                                                 random_state=self.seed)

            # plot the distribution of the continuous columns
            self._plot_stratified_distribution(splits_target={'old_train': old_train.copy(),
                                                              'train': train_df[
                                                                  [self.discrete_column, self.cont_column]].copy(),
                                                              'val': val_df[
                                                                  [self.discrete_column, self.cont_column]].copy(),
                                                              'test': test_df[
                                                                  [self.discrete_column, self.cont_column]].copy()},
                                               compare_old_train=True,
                                               plot_show=plot,
                                               plot_save=save_plot
                                               )
            self._save_indexes_splits(train_df=train_df, val_df=val_df, test_df=test_df)
            return train_df, val_df, test_df

        print(f'{self.dataset[self.discrete_column].value_counts()}')
        ##### Boundaries or Inner domain algorithm
        n_levels = np.sort(self.dataset[self.discrete_column].unique())
        fig, axes = plt.subplots(nrows=1,
                                 ncols=len(n_levels),
                                 figsize=(6 * len(n_levels), 6))

        if self.method in  self._boundary_selection_methods:
            # dataset will be split based on how the classes are distributed in the target
            self.dataset['rows_to_keep'] = np.nan
            # loop through the classes to select the subsets based on each level and plot for latter visualization
            for idx, level_ in enumerate(n_levels):
                indexes = self._select_subset_index(level=level_, method=self.method)
                self.dataset.loc[indexes, 'rows_to_keep'] = 1
                # plot the distribution of the continuous columns
                self._plot_selection(ax=axes[idx],
                                     sliced_column=self.dataset.loc[indexes, self.cont_column],
                                     original_column=self.dataset.loc[self.dataset[self.discrete_column] == level_,
                                     self.cont_column],
                                     level=level_,
                                     )
            # tight and save figures from _plot_selection()
            plt.tight_layout()
            if save_plot:
                plt.savefig(self.output_path.joinpath('DataClass_Dist_Hist_SlicedOriginal.png'), dpi=300)
            if plot:
                plt.show()
            self._close_all_plots()

            # remove the rows that are not included in our selection
            df_reduced = self.dataset.loc[~pd.isna(self.dataset['rows_to_keep']), :].copy()
            df_reduced.drop(columns=['rows_to_keep'],
                            inplace=True)
            original_count = self.dataset.shape[0]
            reduced_count = df_reduced.shape[0]
            reduction_percentage = ((original_count - reduced_count) / original_count) * 100

            # Improved print statement to report dataset reduction in both absolute and relative terms
            print(f'Dataset dimension reduced from {original_count} to {reduced_count} rows, '
                  f'which is a reduction of {reduction_percentage:.2f}%.')
            print(f'{df_reduced[self.discrete_column].value_counts()}')
            # 3. Split selection in the train, validation and test set
            train_df, val_df, test_df = self._split_dataset_train_val_test(
                dataset=df_reduced,
                stratify=self.stratify,
                test_size=.20,
                val_size=.10,
                shuffle=True,
                seed=self.seed
            )
        else:
            # use all the samples of the dataset to perform the split
            train_df, val_df, test_df = self._split_dataset_train_val_test(
                dataset=self.dataset,
                stratify=self.stratify,
                test_size=.20,
                val_size=.10,
                shuffle=True,
                seed=self.seed
            )

        # 4. If we want to have the splits to have equal number of samples for each target group, we clip
        if self.equal_samples_class and self.stratify:
            # reduce the dataset then we stratify on the target
            splits = self._computed_fixed_sample_size([train_df, val_df, test_df])
            train_df = splits[0]
            val_df = splits[1]
            test_df = splits[2]

        self._plot_stratified_distribution(splits_target={
            'train': train_df[
                [self.discrete_column, self.cont_column]].copy(),
            'val': val_df[
                [self.discrete_column, self.cont_column]].copy(),
            'test': test_df[
                [self.discrete_column, self.cont_column]].copy()},
            compare_old_train=False,
            plot_show=plot,
            plot_save=save_plot
        )
        self._save_indexes_splits(train_df=train_df, val_df=val_df, test_df=test_df)
        return train_df, val_df, test_df

    def _save_indexes_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save the indexes of the train, val, and test DataFrames to JSON, selecting the second index if MultiIndex.
        Useful for reproducibility
        :param train_df: pd.DataFrame, DataFrame of the training set
        :param val_df: pd.DataFrame, DataFrame of the validation set
        :param test_df: pd.DataFrame, DataFrame of the test set
        :return: None
        """
        # Helper function to extract index data
        def get_index_data(df):
            if isinstance(df.index, pd.MultiIndex):
                return df.index.get_level_values(1).tolist()  # Extracts the second level index
            else:
                return df.index.tolist()  # Extracts the single level index

        # Create dictionary of indices
        self.indexes_splits = {
            'train': get_index_data(train_df),
            'val': get_index_data(val_df),
            'test': get_index_data(test_df),
        }

        with open(self.output_path.joinpath('SplitIndexes.json'), 'w') as f:
            json.dump(self.indexes_splits, f, indent=4)

    @staticmethod
    def _close_all_plots():
        plt.clf()
        plt.cla()
        plt.close()

    def _select_subset_index(self, level: int, method:str) -> list[int]:
        """
        Implement the selection method using the discrete column to select the area of interest.
        We select either the tails of the distribution of a specific class group or the interior domain
        :param level: int, level of the ordinal target perform the selection
        :return: list of integers to slice on the dataset
        """
        if method in ['boundaries', 'interior_domain']:
            df_subset = self.dataset.loc[self.dataset[self.discrete_column] == level, self.cont_column].sort_values()
            len_select = int((df_subset.shape[0] * self.method_percent) / 2)
            if method == 'boundaries':
                lower_tail = df_subset.iloc[:len_select].index.to_list()
                upper_tail = df_subset.iloc[-len_select:].index.to_list()
                return lower_tail + upper_tail

            elif method == 'interior_domain':
                center = int(df_subset.shape[0] / 2)
                inner_right = df_subset.iloc[center:center + len_select].index.tolist()
                inner_left = df_subset.iloc[max(0, center - len_select):center].index.tolist()
                return inner_right + inner_left
        else:
            raise ValueError(f'Method of selection not available in current method for boundaries or interioir dom')

    @staticmethod
    def _split_dataset_train_val_test(dataset: pd.DataFrame,
                                      stratify: str,
                                      test_size: float,
                                      val_size: float,
                                      seed: int,
                                      shuffle: bool = Tuple,
                                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        partition the dataset into train, validation, and test. No shuffle is applied as the dataset is previously.
        shuffle.
        :param dataset: pd.DataFrame, complete dataframe to partition
        :param stratify: column to stratify the dataset
        :param test_size: test size
        :param val_size: validation size
        :param shuffle: if to shuffle, default = True
        :param seed: random seed
        :return:
        - pd.DataFrames: train, validation, test
        """
        print(f'Data split train, val and test WITH shuffle splits = {shuffle}')

        train_val_df, test_df = train_test_split(dataset,
                                                 test_size=test_size,
                                                 stratify=dataset[stratify],
                                                 shuffle=shuffle,
                                                 random_state=seed)
        # Split the temporary data into equal halves for validation and testing, again stratifying by the
        # 'modality_combo' column
        train_df, val_df = train_test_split(train_val_df,
                                            test_size=val_size,
                                            stratify=train_val_df[stratify],
                                            shuffle=shuffle,
                                            random_state=seed)

        assert train_df.shape[0] + val_df.shape[0] + test_df.shape[0] == dataset.shape[0]

        print(f'\nD_train: {train_df.shape}, D_val: {val_df.shape} , D_test: {test_df.shape}')

        return train_df, val_df, test_df

    def _computed_fixed_sample_size(self,
                                    df_splits: List[pd.DataFrame],
                                    plot: Optional[bool] = False
                                    ) -> List[pd.DataFrame]:
        """
        To avoid class imbalance, we reduce the dataset to balance the classes based on the target discrete class.
        The fixed sample size is calculated from the smallest class in the dataset divided by the number
        of partitions we will have to do.

        :param df_splits: List of DataFrame partitions (e.g., [train_df, val_df, test_df])
        :param plot: bool, if to plot a bar plot comparison on how the discrete target distribution has changed
        :return: List of modified DataFrame partitions with balanced class distributions.
        """
        print('Slicing the splits to contain equal number of samples in all splits')
        modified_splits = []
        dimensions_report = []
        for split_ in df_splits:
            original_size = split_.shape[0]
            # Get the count of the smallest class
            smallest_target_group = split_[self.discrete_column].value_counts().min()
            # Calculate the number of observations per group (class)
            number_groups = split_[self.discrete_column].nunique()
            obsv_per_strata = int(np.floor(smallest_target_group / number_groups))

            # Sample without resetting the index to preserve original indices
            if self.stratify and self.stratify in split_.columns:
                # Using groupby and sample to balance each stratum
                sampled = split_.groupby(self.stratify, group_keys=False).apply(
                    lambda x: x.sample(min(obsv_per_strata, len(x)), random_state=self.seed)
                )
            else:
                # Sampling directly if no stratify column provided
                sampled = split_.sample(n=obsv_per_strata * number_groups, random_state=self.seed)

            modified_splits.append(sampled)
            reduced_size = sampled.shape[0]
            print(f'Reduced size from {original_size} to {reduced_size} in split.')

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
                sns.barplot(data=split_[self.discrete_column].value_counts().to_dict(),
                            color='lightblue',
                            label='Original Samples',
                            ax=ax[0])
                ax[0].set_title('Original Distribution Target')
                ax[0].set_xlabel('Target Class Values In Split')
                ax[0].set_ylabel('Count')
                ax[0].tick_params(axis='x', rotation=45)
                ax[0].grid(True, which='major', linestyle='-', linewidth='0.5',
                           color='grey')
                sns.barplot(data=sampled[self.discrete_column].value_counts().to_dict(),
                            color='orange',  # Corrected color name
                            label='Clipped Samples',
                            ax=ax[1])
                ax[1].set_title('Distribution After Clipping Target')
                ax[1].set_xlabel('Target Class Values In Split')
                ax[1].set_ylabel('Count')
                ax[1].tick_params(axis='x', rotation=45)
                ax[1].grid(True, which='major', linestyle='-', linewidth='0.5',
                           color='grey')
                plt.tight_layout()
                plt.show()
            # Collect data for reporting
            dimensions_report.append([split_.name if hasattr(split_, 'name') else "Unnamed Split",
                                      original_size, reduced_size])

        # Print the dimension changes using tabulate
        print(tabulate(dimensions_report,
                       headers=["Split", "Original Size", "Reduced Size"],
                       tablefmt="grid"))

        return modified_splits

    @staticmethod
    def _split_and_resample_sklearn_samplers(
            x: pd.DataFrame,
            y: pd.Series,
            method='undersample_majority',
            random_state=42) -> pd.DataFrame:

        """
        Applies a resampling method to balance the given set. Ideally should be the training set.
        Supported methods: 'oversample_minority', 'undersample_majority', 'smote', 'ADASYN'.
        Reference: https://towardsdatascience.com/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a
        :param x: pd.DataFrame, complete frame with all columns
        :param y: pd.Series, target column of the frame to apply
        :param method: The resampling method to use on the training set ('oversample_minority', 'undersample_majority',
        'smote', 'ADASYN')
        :param random_state: Random state for reproducibility.
        :return:
        - sampled frame
        """
        # Apply resampling to the training set
        sampler = None
        if method == 'oversample_minority':
            # duplicates existing examples from the minority class with replacements
            sampler = RandomOverSampler(random_state=random_state,
                                        sampling_strategy='not majority',
                                        )
        elif method == 'undersample_majority':
            # removes existing samples from the majority class.
            sampler = RandomUnderSampler(random_state=random_state,
                                         sampling_strategy='not minority',
                                         )
        elif method == 'smote':
            #  Synthetic Minority Over-sampling Technique
            sampler = SMOTE(random_state=random_state,
                            sampling_strategy='not majority',
                            )
        elif method == 'ADASYN':
            sampler = ADASYN(random_state=random_state,
                             sampling_strategy='not majority',
                             )

        x_resampled, _ = sampler.fit_resample(x, y)

        return x_resampled

    def _plot_stratified_distribution(self,
                                      splits_target: dict[str, pd.DataFrame],
                                      compare_old_train: Optional[bool] = False,
                                      plot_show: Optional[bool] = False,
                                      plot_save: Optional[bool] = True
                                      ):
        """
        Plot the stratified target (categorical/ordinal) as a bar plot. The  x axis contains the train, validation, and
        test split. Each x-ticks has the bar of the count of each class in the split
        :return:
        """
        # from dictionary of frames make a unique frame
        concat_frames = []
        for split_name, df in splits_target.items():
            df['split'] = split_name
            concat_frames.append(df)

        concat_frames = pd.concat(concat_frames)
        # palette_ahi = {ahi_int: color for ahi_int, color in zip(concat_frames[self.discrete_column].unique(),
        #                                                         sns.color_palette(n_colors=24))}

        # concat_frames['color'] = concat_frames[self.discrete_column].map(palette_ahi)

        custom_palette = sns.color_palette("rocket_r")

        counts_df = concat_frames[[self.discrete_column, 'split']].groupby(by='split').value_counts().reset_index(
            drop=False)  # .sort_values(by=self.discrete_column)
        counts_df = counts_df.sort_values(by=['split', 'AHI_multiclass'])
        max_target = concat_frames[self.cont_column].max()
        if compare_old_train:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1])  # Updated width_ratios to match the number of columns

            # Create the left column plot (spanning first two rows)
            ax0 = fig.add_subplot(gs[0:2, 0])
            sns.barplot(data=counts_df.loc[counts_df['split'].isin(['old_train', 'train'])],
                        x=self.discrete_column,
                        y='count',
                        ax=ax0,
                        palette=['blueviolet', 'salmon'],
                        hue='split')
            ax0.set_title('Original vs Re-sampled')
            ax0.legend()

            # Create the center column plots
            ax_train_bar = fig.add_subplot(gs[0, 1])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'train'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_train_bar,
                        palette=custom_palette
                        )
            ax_train_bar.grid(alpha=0.7, axis='y')
            ax_train_bar.set_title(f"Train - Dim: {counts_df.loc[counts_df['split'] == 'train'].sum()['count']}")

            ax_val_bar = fig.add_subplot(gs[1, 1])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'val'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_val_bar,
                        palette=custom_palette
                        )
            ax_val_bar.grid(alpha=0.7, axis='y')

            ax_val_bar.set_title(f"Validation - Dim: {counts_df.loc[counts_df['split'] == 'val'].sum()['count']}")

            ax_test_bar = fig.add_subplot(gs[2, 1])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'test'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_test_bar,
                        palette=custom_palette
                        )
            ax_test_bar.grid(alpha=0.7, axis='y')
            ax_test_bar.set_title(f"Test - Dim: {counts_df.loc[counts_df['split'] == 'test'].sum()['count']}")

            ax_train_dist = fig.add_subplot(gs[0, 2])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'old_train', self.cont_column],
                        color='blueviolet',
                        fill=True,
                        common_grid=True,
                        ax=ax_train_dist,
                        label='old train',
                        )
            # ax_train_dist.legend()
            ax_train_dist.set_xlim([0, max_target])
            ax_train_dist.set_title('Train Distribution')

            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'train', self.cont_column],
                        color='salmon',
                        fill=True,
                        label='train',
                        ax=ax_train_dist
                        )
            ax_train_dist.set_xlim([0, max_target])
            ax_train_dist.legend()

            ax_val_dist = fig.add_subplot(gs[1, 2])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'val', self.cont_column],
                        color='salmon',
                        fill=True,
                        ax=ax_val_dist)
            # ax_val_dist.legend()
            ax_val_dist.set_xlim([0, max_target])
            ax_val_dist.set_title('Validation Distribution')

            ax_test_dist = fig.add_subplot(gs[2, 2])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'test', self.cont_column],
                        color='salmon',
                        fill=True,
                        ax=ax_test_dist
                        )
            # ax_test_dist.legend()
            ax_test_dist.set_title('Test Distribution')
            ax_test_dist.set_xlim([0, max_target])
            plt.tight_layout()
            if plot_save:
                plt.savefig(self.output_path.joinpath('DataClass_Distribution_BarKde_OldTrainNewTrain.png'), dpi=300)
            if plot_show:
                plt.show()
            self._close_all_plots()

        else:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])  # Updated width_ratios to match the number of columns

            # Create the center column plots
            ax_train_bar = fig.add_subplot(gs[0, 0])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'train'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_train_bar,
                        palette=custom_palette
                        )
            ax_train_bar.grid(alpha=0.7, axis='y')
            ax_train_bar.set_title(f"Validation - Dim: {counts_df.loc[counts_df['split'] == 'train'].sum()['count']}")

            ax_val_bar = fig.add_subplot(gs[1, 0])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'val'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_val_bar,
                        palette=custom_palette
                        )
            ax_val_bar.grid(alpha=0.7, axis='y')
            ax_val_bar.set_title(f"Validation - Dim: {counts_df.loc[counts_df['split'] == 'val'].sum()['count']}")

            ax_test_bar = fig.add_subplot(gs[2, 0])
            sns.barplot(data=counts_df.loc[counts_df['split'] == 'test'],
                        x=self.discrete_column,
                        y='count',
                        ax=ax_test_bar,
                        palette=custom_palette
                        )
            ax_test_bar.grid(alpha=0.7, axis='y')
            ax_test_bar.set_title(f"Validation - Dim: {counts_df.loc[counts_df['split'] == 'test'].sum()['count']}")

            ax_train_dist = fig.add_subplot(gs[0, 1])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'train', self.cont_column],
                        color='salmon',
                        fill=True,
                        label='train',
                        ax=ax_train_dist
                        )
            ax_train_dist.set_xlim([0, max_target])
            ax_train_dist.legend()

            ax_val_dist = fig.add_subplot(gs[1, 1])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'val', self.cont_column],
                        color='salmon',
                        fill=True,
                        ax=ax_val_dist)
            # ax_val_dist.legend()
            ax_val_dist.set_xlim([0, max_target])
            ax_val_dist.set_title('Validation Distribution')

            ax_test_dist = fig.add_subplot(gs[2, 1])
            sns.kdeplot(data=concat_frames.loc[concat_frames['split'] == 'test', self.cont_column],
                        color='salmon',
                        fill=True,
                        ax=ax_test_dist
                        )
            # ax_test_dist.legend()
            ax_test_dist.set_title('Test Distribution')
            ax_test_dist.set_xlim([0, max_target])
            plt.tight_layout()
            if plot_save:
                plt.savefig(self.output_path.joinpath('DataClass_Distribution_BarKde.png'), dpi=300)
            if plot_show:
                plt.show()
            self._close_all_plots()
    def _plot_selection(self,
                        sliced_column: pd.Series,
                        original_column: pd.Series,
                        level: int,
                        ax: Axes = None):
        """
        To plot all the figures in a single main figure we receive an axis where all the figures will be located.
        The function plots how each level id distributed as a histogram and overimposed by the selected boundaries
        :param ax: axis to allocate each plot
        :param sliced_column: selected column at the given level by the boundary or inner domain algorithm
        :param original_column: original level distribution
        :param level: category in the target columns
        :return:
        """
        if ax is not None:
            sns.histplot(original_column, kde=False, stat='frequency', label=f'Original - Class {level}',
                         element='step', ax=ax)
            sns.histplot(sliced_column, kde=False, stat='frequency',
                         label=f'Sliced - {self.method} {self.method_percent*100}%',
                         element='step', alpha=.5, ax=ax)

            # Plot mean and median lines
            mean_value = sliced_column.mean()
            median_value = sliced_column.median()
            ax.axvline(mean_value, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(median_value, color='blue', linestyle='--', linewidth=1.5)
            ax.legend(title='Target Class')
            ax.set_title(f'Distribution of {self.cont_column} by Target Class {level} with Mean and Median')
            ax.set_xlabel(self.cont_column)
            ax.set_ylabel('Frequency')
            ax.grid(True)
        else:
            # plot one by one in different figures
            plt.figure(figsize=(12, 8))
            sns.histplot(original_column,
                         kde=False,
                         stat='frequency',
                         label=f'Original - Class {level}',
                         element='step')

            sns.histplot(sliced_column,
                         kde=False,
                         stat='frequency',
                         label=f'Sliced - {self.method} {self.method_percent*100}%',
                         element='step',
                         alpha=.5)
            # Plot mean and median lines
            mean_value = sliced_column.mean()
            median_value = sliced_column.median()
            plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5)
            plt.axvline(median_value, color='blue', linestyle='--', linewidth=1.5)
            plt.legend(title='Target Class')
            plt.title('Distribution of AHI by Target Class with Mean and Median')
            plt.xlabel('AHI')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            # if plot_save:
            #     plt.savefig(self.output_path.joinpath('DataClass_Dist_Hist_SlicedOriginal.png'),  dpi=300)
            # if plot_show:
            plt.show()
# %%%

# class DatasetReady(ahi_class):
#     def __init__(self,
#                  dataset: pd.DataFrame,
#                  val_size: float,
#                  train_size: float,
#                  test_size: float,
#                  target:str,
#                  stratify: Optional[Union[str, None]] = None,
#                  learning_task: str = 'regression',
#                  num_classes: Optional[Union[int, None]] = None,
#                  equal_samples_class: bool = False,
#                  random_state:int=42,
#                  ):
#         """
#         :param val_size: float, val size 0< val_size <1
#         :param train_size: float, 0< train_size <1
#         :param test_size: float, 0< test_size <1
#         :param stratify: str, column to stratify the data
#         :param fixed_test_samples: if not None,  defined a number of samples in the test set
#         :param num_classes: how many classes on the target (2 , 3 or 4)
#         :param fixed_test_samples:
#         """
#         self.dataset = dataset.copy()
#         self.stratify = stratify
#         self.learning_task = learning_task
#         self.target = target
#         self.num_classes = num_classes
#         self._validate_input()
#         self.target_discrete = f'{target}_discrete'
#
#         # Partitions variables
#         if isinstance(train_size, float) and isinstance(val_size, float) and isinstance(test_size, float):
#             self.train_val_test_split = True
#         elif isinstance(train_size, float) and isinstance(val_size, float) and test_size is None:
#             self.train_val_test_split = False
#         else:
#             raise ValueError(f'Define properly the data splits')
#         self.val_size = val_size
#         self.train_size = train_size
#         self.test_size = test_size
#         self.equal_samples_class = equal_samples_class
#         self.seed = random_state
#
#         # the regression_bool variables disables the stratified option in the data splits
#         self.regression_bool = False
#
#         if not stratify:
#             self.asq_dataset = shuffle(self.asq_dataset, random_state=self.config['seed'])
#             self.shuffle_plits = False
#         else:
#             self.shuffle_plits = True
#         print('Dataset extracted and ready for partition, please run method .logic_pipeline() to compute the actions')
#
#     def _validate_input(self):
#         """Validate the inputs"""
#         if not self.learning_task in ['classification', 'regression']:
#             warnings.warn(f"Learning task must be  'classification' or 'regression' ")
#
#         if self.learning_task == 'regression' and not self.num_classes is None and not self.stratify:
#             warnings.warn(f"Number of classes must be None when regression and not stratification based "
#                           f"on the target")
#
#         if not self.target in self.dataset.columns:
#             warnings.warn(f'Target {self.target} is not in the columns of the dataset')
#
#         if not self.num_classes in [2,3,4]:
#             warnings.warn(f'Number of classes must be one of the following values [2,3,4]')
#
#     # %% called function
#     def partition_data_train_val_tests(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         partition the dataset into train, validation, and test. No shuffle is applied as the dataset is previously.
#         shuffle.
#         :return:
#         """
#         if self.equal_samples_class and self.stratify == self.target:
#             # reduce the dataset then we stratify on the target
#             self._computed_fixed_sample_size()
#
#         print(f'Data split train, val and test WITH shuffle splits = {self.shuffle_plits}')
#
#         train_val_df, test_df = train_test_split(self.dataset,
#                                                  test_size=self.test_size,
#                                                  stratify=self.stratify,
#                                                  shuffle=self.shuffle_plits,
#                                                  random_state=self.seed)
#         # Split the temporary data into equal halves for validation and testing, again stratifying by the
#         # 'modality_combo' column
#         train_df, val_df = train_test_split(train_val_df,
#                                             test_size=self.val_size,
#                                             stratify=self.stratify,
#                                             shuffle=self.shuffle_plits,
#                                             random_state=self.seed)
#
#         assert train_df.shape[0] + val_df.shape[0] + test_df.shape[0] == self.dataset.shape[0]
#
#         print(f'\nD_train: {train_df.shape}, D_val: {val_df.shape} , D_test: {test_df.shape}')
#
#         if self.regression_bool:
#             # because regression problem, after the stratified split we pass from multiclass to continues
#             for splits_ in [train_df, val_df, test_df]:
#                 splits_.loc[:, self.target] = self.target_continues.loc[splits_.index, self.target]
#
#         return train_df, val_df, test_df
#
#     # %% getters
#     def get_features(self) -> list:
#         """Return the initial features from the complete datase (excluded ID, and traget)"""
#         return self.dataset.columns.tolist()
#
#     def get_complete_dataset(self) -> pd.DataFrame:
#         """Return the complete dataset"""
#         return self.dataset
#
#     # %% hidden methods
#     def _create_discrete_target(self):
#         if self.num_classes and self.stratify == self.target and not (self.target_discrete in self.dataset):
#             # stratify on the target
#             self._store_continues_target()
#             if 'log' in self.target.lower():
#                 # if the target is in the log +1 scale, transform to the normal scale
#                 self.dataset[self.target_discrete] = self.dataset[self.target].apply(lambda x: np.exp(x) - 1)
#                 print(f'Target transformed from the log(x + 1) scale to x scale')
#                 self.dataset[self.target_discrete] = self.ahi_class(x=self.dataset[self.target],
#                                                            num_class=self.num_classes).values
#
#     def _computed_fixed_sample_size(self):
#         """
#         To avoid class imbalance we reduce teh datsaet to balance teh classes based on the target discrte class
#         The fixed sample size is calculated from the smallest class in the dataset divided by the number
#         of partitions we will have to do.
#         """
#         shape_initial_data = self.dataset.shape[0]
#         if not self.target_discrete in self.dataset.columns:
#             # this means the stratification is on the discrete target
#             self._create_discrete_target()
#             stratify = self.target_discrete
#         else:
#             # the stratification and balance class is on another variable
#             stratify = self.stratify
#         smallest_target_group = self.dataset[stratify].value_counts().min()
#         # Sampling the same number of instances from each class
#         self.dataset = self.dataset.groupby(stratify).apply(lambda x: x.sample(smallest_target_group,
#                                                                                   random_state=self.seed)
#                                                             ).reset_index(drop=True)
#         print(f'Dataset shrunk from {shape_initial_data} to {self.dataset}')
#
#
#     def _store_continues_target(self) :
#         print(f'Continuous target stores in class DatasetReady, variable self.target_continues')
#         self.target_continues = self.dataset[self.target].copy()
#
#     def plot_target_distribution_on_splits(self, data: dict, plot_path: pathlib.Path):
#         """
#         Plot the target distribution on the different split to heck the clas imbalance
#         :param data:
#         :return:
#         """
#         custom_palette = sns.color_palette("rocket_r")
#         fontsize_axis = 20
#         fontsize_title = 20
#         # data = {
#         #     'train': D_train.ahi,
#         #     'val': D_val.ahi,
#         #     'test': D_test.ahi,
#         # }
#         # max_len = max( [max(data[key].value_counts()) for key in [*data.keys()]]) +50
#         if self.regression_bool:
#             sns.set_context("paper", rc={"font.size": fontsize_axis, "axes.titlesize": fontsize_title,
#                                          "axes.labelsize": fontsize_axis})
#             fig, axs = plt.subplots(nrows=len(data), ncols=1, figsize=(8, 11))
#             fig.suptitle("Target Distribution log(AHI+1)", fontsize=fontsize_title)
#             for ax_, key_ in zip(axs, data.keys()):
#                 sns.histplot(x=key_, data=data,
#                              hue_order=['os1', 'osa2', 'osa3', 'osa4'],
#                              palette=custom_palette,
#                              ax=ax_
#                              )
#                 ax_.set_ylim([0, max(data[key_].value_counts()) + 100])  # max_len
#                 ax_.grid()
#                 # ax_.set_xticklabels(lbls, fontsize=fontsize_axis)
#                 ax_.title.set_text(f"{key_.capitalize()}, Samples:{data[key_].shape[0]}")
#
#             plt.tight_layout(pad=1.8)
#             plt.show()
#             plt.draw()
#
#         else:
#             num_classes = [*data['test'].unique()]
#             num_classes.sort()
#             if len(num_classes) == 4:
#                 lbls = ['Normal', 'Mild', 'Moderate', 'Severe']
#             elif len(num_classes) == 2:
#                 lbls = ['Normal_Mild', 'Moderate_Severe']
#             elif len(num_classes) == 3:
#                 lbls = ['Normal_Mild', 'Moderate', 'Severe']
#
#             sns.set_context("paper", rc={"font.size": fontsize_axis, "axes.titlesize": fontsize_title,
#                                          "axes.labelsize": fontsize_axis})
#             fig, axs = plt.subplots(nrows=len(data), ncols=1, figsize=(8, 11))
#             fig.suptitle("Target Distribution On Different Splits ", fontsize=fontsize_title)
#             for ax_, key_ in zip(axs, data.keys()):
#                 sns.countplot(x=key_, data=data,
#                               hue_order=['os1', 'osa2', 'osa3', 'osa4'],
#                               palette=custom_palette,
#                               ax=ax_
#                               )
#                 ax_.set_ylim([0, max(data[key_].value_counts()) + 100])  # max_len
#                 ax_.grid()
#                 ax_.set_xticklabels(lbls, fontsize=fontsize_axis)
#                 ax_.title.set_text(f"{key_.capitalize()}, Samples:{data[key_].shape[0]}")
#
#             plt.tight_layout()
#             plt.show()
#             plt.draw()
#         fig.savefig(plot_path / 'target_distribution_splits.png', dpi=300)
#

# if __name__ == "__main__":
#     pass
