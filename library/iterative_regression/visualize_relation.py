"""
Visualize the relationship between the target and the responses from a questionnaire.

The function works when we the target is cotnunous, if the target is multi class then we should plot the logits in the
logarithm scale as we intend to use linear regression or logistic regression as model.
"""
import pathlib
import pandas as pd
from typing import Union, Optional
from config.config import config, sections
import seaborn as sns
import matplotlib.pyplot as plt
# from utils import get_ordinal_encoding_log
from tqdm import tqdm


class VisualizeRelation:
    def __init__(self,
                dataset: pd.DataFrame,
                dep_var: Union[str, list],
                indep_var:list,
                save_path:pathlib.Path,
                responses_labels:dict
                ):
        """
        Generate bar plot and box plot of the questionnaire where we visualize the response distribution against the
        target of each question
        :param dataset: pd.Dataframe, dataset with the questionnaire
        :param dep_var: Union[str, list], features names in the dataset
        :param indep_var: list, target column name in the dataset that we will plot in y-axis
        :param save_path: pathlib.Path, path of where to save the plots
        :param responses_labels: dict, of the xticks, we have the mapping of numerical to labels of the responses, so
        we can know what does each response means. They keys must match with the column names for them to be utilized.
        """
        self.data = dataset
        self.dep_var = dep_var
        self.indep_var = indep_var
        self.save_path = save_path
        self.x_ticks_labels = responses_labels

    def run(self):
        if isinstance(self.dep_var, list):
            for target_ in self.dep_var:
                self._compute_visualization(target=target_)
        else:
            self._compute_visualization(target=self.dep_var)

    def _compute_visualization(self,
                               target:str):
        for indep_var in tqdm(self.indep_var,
                              desc="Generating Plots"):

            type_ = self._is_what_type(col=indep_var)
            # if type_ == 'binary':
            #     self._plot_binary_relation(column=indep_var,
            #                                target=target)
            if type_ == 'multi_class' or type_ == 'binary':
                self._plot_multi_class_relation(column=indep_var,
                                                target=target)
            elif type_ == 'continuous':
                self._plot_continuous_relation(column=indep_var,
                                           target=target)

            else:
                print(f'Undefined type for independent variable {indep_var}')

            self._generate_csv_response_count(column=indep_var)
    def _is_what_type(self,
                      col: str) -> str:
        unique_count = self.data[col].dropna().nunique()
        if unique_count == 2:
            return 'binary'
        elif 2 < unique_count < 9:
            return 'multi_class'
        else:
            return 'continuous'

    def _plot_continuous_relation(self, column: str,
                                  target:str):
        if column in self.x_ticks_labels.keys():
            xticks = self.x_ticks_labels.get(column)
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data.loc[~self.data[column].isna()],
                        x=column,
                        y=target)
        sns.regplot(data=self.data.loc[~self.data[column].isna()],
                    x=column,
                    y=target,
                    scatter=False,
                    color='red',
                    line_kws={"alpha": 0.5})
        plt.title(f'{column} vs {target}')
        if column in self.x_ticks_labels.keys():
            plt.xticks(ticks=xticks.keys(),
                       labels=xticks.values())
        plt.xlabel(column)
        plt.ylabel(target)
        plt.tight_layout()
        plt.grid(alpha=0.7)
        plt.savefig(self.save_path / f'{column}_vs_{target}_scatter.png', dpi=300)
        plt.close()

    def _plot_multi_class_relation(self,
                                   column: str,
                                   target: str):
        """
        Create a figure with two subplots (1 row, 2 columns) where we have a box plot showing how the target is
        distributed among the responses. Whereas, the left presents a bar plot of how many people answered the response
        :param column:
        :param target:
        :return:
        """
        data_plot = self.data.loc[~self.data[column].isna() &
                                  (~self.data[target].isna())]
        if column in self.x_ticks_labels.keys():
            xticks = self.x_ticks_labels.get(column)

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle(f'{column}\n Observations: {data_plot.shape[0]}', fontsize=16)
        # Box plot on the left
        sns.boxplot(data=data_plot, x=column, y=target, ax=axes[0])
        axes[0].set_title(f'Distribution Of Target On Each Response')
        if column in self.x_ticks_labels.keys():
            axes[0].set_xticks(ticks=[xticks[label] for label in xticks.keys()])
            axes[0].set_xticklabels(labels=xticks.keys())
        axes[0].set_xlabel(column)
        axes[0].set_ylabel(target)
        axes[0].grid(alpha=0.7)

        # Count of each column unique value on the right
        sns.countplot(data=data_plot, x=column, ax=axes[1])
        axes[1].set_title(f'Count of Responses')
        if column in self.x_ticks_labels.keys():
            axes[1].set_xticks(ticks=[xticks[label] for label in xticks.keys()])
            axes[1].set_xticklabels(labels=xticks.keys())
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('Count')
        axes[1].grid(alpha=0.7)
        # Adding count on top of each bar
        for p in axes[1].patches:
            axes[1].annotate(format(p.get_height(), '.0f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points')
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.save_path / f'{column}_vs_{target}_multiclass.png', dpi=300)
        plt.close()


    def _plot_binary_relation(self,
                              column: str,
                              target:str):
        if column in self.x_ticks_labels.keys():
            xlbls = self.x_ticks_labels.get(column)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.data.loc[~self.data[column].isna()],
                    x=column,
                    y=target)
        # sns.scatterplot(data=self.data.loc[~self.data[column].isna()],
        #               x=column,
        #               y=target)
        # sns.regplot(data=self.data.loc[~self.data[column].isna()],
        #             x=column,
        #             y=target,
        #             scatter=False,
        #             color='red',
        #             line_kws={"alpha": 0.5})
        plt.title(f'{column} vs {target}')
        if column in self.x_ticks_labels.keys():
            plt.xticks(ticks=xlbls.keys(),
                       labels=xlbls.values())
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title=target)
        plt.tight_layout()
        plt.grid(alpha=0.7)
        plt.savefig(self.save_path / f'{column}_vs_{target}_binary.png', dpi=300)
        plt.close()

    def _generate_csv_response_count(self, column:str):
        """
        Generate a csv where we track the variable name, number of responses on each category and total number of
        observations on each
        :param column:
        :return:
        """
        data_plot = self.data.loc[~self.data[column].isna()]
        self.x_ticks_labels

        pass

def get_figure(variable: str,
               file_path: Optional[pathlib.Path]=None):
    """
    Search for png images already saved in the directory
    :param variable: str, variable name which is found in the first substring of the file.png
    :param file_path: Optional[pathlib.Path] path where the images are located
    :return:
    """
    if file_path is None:
        file_path = config.get('results_relation_plots')
    for file in file_path.glob('**/*.png'):
        current_file_name = file.name
        current_file_name = current_file_name.split('_')[0]
        if current_file_name == variable:
            file_path = file
            break
    # If file found, display the image
    if file_path:
        img = plt.imread(file_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"No figure found for column '{variable}' in the given path.")























