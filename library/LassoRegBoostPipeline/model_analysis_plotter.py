"""
This class is made to plot they ket features to evaluate the performance of a selected model given the path of the
model.
"""
import pathlib
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from LassoRegBoostPipeline.utils import EffectMeasurePlot

class ModelAnalysisPlotter:
    def __init__(self, model_path: Union[str, pathlib.Path]):
        if isinstance(model_path, str):
            self.model_path = pathlib.Path(model_path)
        else:
            self.model_path = model_path
        self.name_model = self.model_path.name.split('-')[3][2:]

    def _plot_model_analysis(self, sub_string: Optional[str] = None,
                             figsize: Optional[Tuple[int, int]] = (14, 12)):
        """
        Plot various analysis plots for a given model
        :param sub_string: str, to add to the title in case we want to be more clear on the model we are plotting
        :param figsize: tuple[int, int], figure size of the figure
        :return: None
        """
        cm_test = self.model_path.joinpath('confusion_matrix_test.png')
        cm_train = self.model_path.joinpath('confusion_matrix_train.png')
        dist_target = self.model_path.joinpath('Distribution_Model.png')
        train_val_curve = self.model_path.joinpath('train_val_curve.png')

        # Create a figure with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        # Plot first image in the first subplot
        img = mpimg.imread(cm_train)
        axs[0, 0].imshow(img)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Train Confusion Matrix')

        # Plot second image in the second subplot
        img = mpimg.imread(cm_test)
        axs[0, 1].imshow(img)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Test Confusion Matrix')

        # Plot third image in the third subplot
        img = mpimg.imread(dist_target)
        axs[1, 0].imshow(img)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Target Distribution')

        # Plot fourth image in the fourth subplot
        img = mpimg.imread(train_val_curve)
        axs[1, 1].imshow(img)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Training & Validation Curve')

        # Set global title
        if sub_string is not None:
            fig.suptitle(self.name_model + f' {sub_string}')
        else:
            fig.suptitle(self.name_model)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def _plot_regression_results(self, figsize: Optional[Tuple[int, int]] = (14, 16)):
        """
        From the GolbOutput.xlsx table we will plot the odds ratio of the last model used in the regression.
        If the OR plots already exist, it will display them, else, the plot will be created and displayed from the
        global output excel file
        :param figsize: figure size to display the image
        :return: None
        """
        output_path = self.model_path.joinpath('OlsFeatSelc', 'publish', 'or_plot.png')

        if output_path.exists():
            img = mpimg.imread(output_path)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            global_output_path = self.model_path.joinpath('OlsFeatSelc', 'iter', 'GlobOutput.xlsx')
            df_global = pd.read_excel(global_output_path)
            df_last_model = df_global.loc[df_global['model'] == df_global.model.unique()[-1],
                                          ['variable', 'odds', 'odds_ci_low', 'odds_ci_high', 'P>|t|']]
            df_last_model = df_last_model.loc[df_last_model.variable != 'const', :]

            forest_plot = EffectMeasurePlot(
                label=df_last_model.variable.tolist(),
                effect_measure=df_last_model.odds.tolist(),
                lcl=df_last_model.odds_ci_low.tolist(),
                ucl=df_last_model.odds_ci_high.tolist(),
                p_value=df_last_model['P>|t|'].tolist(),
                alpha=0.05
            )
            forest_plot.plot(figsize=figsize, path_save=None, show=True)

    def explore_model(self, sub_string: Optional[str] = None,
                      figsize_multi_plot: Optional[Tuple[int, int]] = (14, 12),  figsize_or_plot: Optional[Tuple[int, int]] = (14, 16)):
        """
        Plot various analysis plots for a given model
        :param sub_string: str, to add to the title in case we want to be more clear on the model we are plotting
        :param figsize_multi_plot: tuple[int, int], figure size of the figure for plot_model_analysis
        :param figsize_or_plot: tuple[int, int], figure size for _plot_regression_results
        :return: None
        """
        self._plot_model_analysis(sub_string=sub_string, figsize=figsize_multi_plot)
        self._plot_regression_results(figsize=figsize_or_plot)


# Example usage
# best_train_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['train_F1 Score'].max(), :]
# plotter = ModelAnalysisPlotter(model_path=best_train_metric['model_path'].values[0])
# plotter.explore_model(sub_string='best train metrics')
#
# best_test_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['test_F1 Score'].max(), :]
# plotter = ModelAnalysisPlotter(model_path=best_test_metric['model_path'].values[0])
# plotter.plot_model_analysis(sub_string='best test metrics')
