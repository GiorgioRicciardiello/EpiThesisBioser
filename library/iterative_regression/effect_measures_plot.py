import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pathlib
import matplotlib.lines as mlines
from typing import Optional, Tuple

class EffectMeasurePlot:
    def __init__(self, label: list[str],
                 effect_measure: list[float, int],
                 lcl: list[float, int],
                 ucl: list[float, int],
                 p_value: list[float, int],
                 alpha:float=0.05):
        """

        :param label: list, variables name
        :param effect_measure: list,
        :param lcl:
        :param ucl:
        :param p_value:
        """
        self.df = pd.DataFrame({
            'study': label,
            'OR': effect_measure,
            'LCL': lcl,
            'UCL': ucl,
            'pvalue': p_value
        })
        self.df['OR2'] = pd.to_numeric(self.df['OR'], errors='coerce')
        self.df['LCL_dif'] = np.abs(self.df['OR2'] - pd.to_numeric(self.df['LCL'], errors='coerce'))
        self.df['UCL_dif'] =  np.abs(pd.to_numeric(self.df['UCL'], errors='coerce') - self.df['OR2'])
        self.em = 'OR'
        self.alpha = alpha
        self.ci = f'{np.round((1 - self.alpha) * 100, 2)}% CI'
        self.scale = 'linear'
        self.center = 1
        self.errc = 'dimgrey'
        self.shape = "o" # 'd'
        self.pc = 'k'
        self.linec = 'gray'

    def set_labels(self, **kwargs):
        self.em = kwargs.get('effectmeasure', self.em)
        self.ci = kwargs.get('conf_int', self.ci)
        self.scale = kwargs.get('scale', self.scale)
        self.center = kwargs.get('center', self.center)

    def set_colors(self, **kwargs):
        self.errc = kwargs.get('errorbarcolor', self.errc)
        self.shape = kwargs.get('pointshape', self.shape)
        self.linec = kwargs.get('linecolor', self.linec)
        self.pc = kwargs.get('pointcolor', self.pc)

    @staticmethod
    def _check_statistical_significance(odd_ratio: float,
                                        lcl: float,
                                        ucl: float,
                                        pvalue: float,
                                        alpha: float) -> bool:
        if odd_ratio > 1 and (lcl > 1 or ucl > 1) and pvalue < alpha:
            # return "Significant (CI does not contain 1)"
            return True
        elif odd_ratio < 1 and (lcl < 1 or ucl < 1) and pvalue < alpha:
            # return "Significant (CI does not contain 1)"
            return True
        else:
            # return "Not Significant (CI contains 1)"
            return False
    def plot(self,
             figsize: Tuple[int, int] = (14, 16),
             t_adjuster: float = 0.01,
             decimal: int = 3,
             size: int = 3,
             max_value: Optional[float] = None,
             min_value: Optional[float] = None,
             text_size: int = 12,
             hline_thickness: float = 0.8,
             color_significant:str='green',
             show:Optional[bool] = False,
             path_save:pathlib.Path=None):
        """
        Plot the effect measures.

        Iterate over the dataset and construct the columns that we see in the plot. This plot identified which OR are
        significant and which one are not. Significant OR are colored in green, and non-significant are colored in
        black. It include the plot of the OR, the OR value, 95%CI and the p-valye

        :param figsize: Figure size.
        :param t_adjuster: Adjustment for the table.
        :param decimal: Decimal precision for values.
        :param size: Size of markers.
        :param max_value: Maximum value for x-axis.
        :param min_value: Minimum value for x-axis.
        :param text_size: Size of text.
        :param hline_thickness: Thickness of horizontal lines.
        :param color_significant: color of the significant OR in the tree plot
        :return: Plot object.
        """
        # create the list with the metrics and mark the y-ticks
        ytick, tval = [], []
        for i, row in self.df.iterrows():
            if not np.isnan(row['OR2']):
                or_ = round(row['OR2'], decimal)
                lcl_ = round(row['LCL'], decimal)
                ucl_ = round(row['UCL'], decimal)
                pval_ = round(row.pvalue, 6)
                # tval.append([or_,  f"({lcl_}, {ucl_}) - {pval_}"])
                tval.append([or_, f"{lcl_} , {ucl_}", pval_])
                ytick.append(i)
            else:
                tval.append([' ', ' '])
                ytick.append(i)
        # set the x axis limits
        if max_value is None:
            maxi = self.df['UCL'].max() + self.df['UCL'].std()/2
            maxi = np.round(maxi, 1)
        else:
            maxi = max_value
        if min_value is None:
            mini = self.df['LCL'].min() - self.df['LCL'].std()
            mini = np.round(mini, 1)
        else:
            mini = min_value

        plt.figure(figsize=figsize)
        gspec = gridspec.GridSpec(1, 6)
        plot = plt.subplot(gspec[0, 0:4])
        tabl = plt.subplot(gspec[0, 4:])
        plot.set_ylim(-1, len(self.df))
        if self.scale == 'log':
            plot.set_xscale('log')
        plot.axvline(self.center,
                     # color=self.linec,
                     zorder=1,
                     linestyle='--',
                     color='orange',
                     alpha=0.5)

        # define the legends we want to show
        significant_legend_handle = mlines.Line2D([], [],
                                                  color=color_significant,
                                                  marker=self.shape,
                                                  markersize=8,
                                                  label='significant')
        non_significant_legend_handle = mlines.Line2D([], [],
                                                      color=self.pc,
                                                      marker=self.shape,
                                                      markersize=8,
                                                      label='Non-significant')
        risk_factor_legend_handle = mlines.Line2D([], [],
                                                  color='salmon',
                                                  linestyle='dashdot',
                                                  linewidth=0.5,
                                                  label='Risk Factor')
        protective_legend_handle = mlines.Line2D([], [],
                                                 color='royalblue',
                                                 linestyle='dashdot',
                                                 linewidth=0.5,
                                                 label='Protective')
        # iteraite over the different values to place in the y-axis of the image
        for i, row in self.df.iterrows():
            i = int(i)
            if not np.isnan(row['OR2']):
                or_ = row['OR2']
                lcl_ = row['LCL']
                ucl_ = row['UCL']
                pval_ = row.pvalue

                if or_ > 1:
                    plot.hlines(y=i,
                                xmin=mini,
                                xmax=mini+0.1,
                                colors='salmon',
                                linestyles='dashdot',
                                linewidths=hline_thickness)
                else:
                    plot.hlines(y=i,
                                xmin=mini,
                                xmax=mini+0.1,
                                linestyles='dashdot',
                                colors='royalblue',
                                linewidths=hline_thickness)

                if self._check_statistical_significance(odd_ratio=or_,
                                                        lcl=lcl_,
                                                        ucl=ucl_,
                                                        pvalue=pval_,
                                                        alpha=self.alpha):
                    # Color the error bars and dots green for significant values
                    plot.errorbar(x=or_,
                                  y=i,
                                  xerr=[[row['LCL_dif']], [row['UCL_dif']]],
                                  marker='None',
                                  zorder=2,
                                  ecolor=color_significant,  # Color for error bars
                                  elinewidth=(size / size),
                                  linewidth=0)
                    plot.scatter(x=or_,
                                 y=i,
                                 c=color_significant,  # Color for dots
                                 s=(size * 15),
                                 marker=self.shape,
                                 zorder=3,
                                 edgecolors='None',
                                 label='significant')

                else:
                    plot.errorbar(x=row['OR2'],
                                  y=i,
                                  xerr=[[row['LCL_dif']], [row['UCL_dif']]],
                                  marker='None',
                                  zorder=2,
                                  ecolor=self.errc,
                                  elinewidth=(size / size),
                                  linewidth=0)
                    plot.scatter(x=or_,
                                 y=i,
                                 c=self.pc,
                                 s=(size * 15),
                                 marker=self.shape,
                                 zorder=3,
                                 edgecolors='None',
                                 label='Non-significant')

        plot.legend(handles=[significant_legend_handle,
                             non_significant_legend_handle,
                             risk_factor_legend_handle,
                             protective_legend_handle],
                    loc='upper right')
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        plot.set_yticks(ytick)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi])
        plot.set_xticklabels([mini, self.center, maxi])
        plot.set_yticklabels(self.df['study'])
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()
        tb = tabl.table(cellText=tval,
                        cellLoc='center',
                        loc='right',
                        colLabels=[self.em, self.ci, 'p-value'],
                        bbox=[0, t_adjuster, 1, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(text_size)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)
        plot.grid(axis='y',
                  alpha=0.7)
        plt.tight_layout()
        if path_save:
            plt.savefig(path_save, dpi=300)
        if show:
            plt.show()
        return plot

    def _original_plot(self,
             figsize: tuple = (12, 12),
             t_adjuster: float = 0.01,
             decimal: int = 3,
             size: int = 3,
             max_value=None,
             min_value=None,
             text_size: int = 12):
        """
        Iterate over the dataset and construct the columns that we see in the plot
        :param figsize:
        :param t_adjuster:
        :param decimal:
        :param size:
        :param max_value:
        :param min_value:
        :param text_size:
        :return:
        """
        # create the list with the metrics and mark the y-ticks
        ytick, tval = [], []
        for i, row in self.df.iterrows():
            if not np.isnan(row['OR2']):
                or_ = round(row['OR2'], decimal)
                lcl_ = round(row['LCL'], decimal)
                ucl_ = round(row['UCL'], decimal)
                pval_ = round(row.pvalue, 6)
                # tval.append([or_,  f"({lcl_}, {ucl_}) - {pval_}"])
                tval.append([or_, f"{lcl_} , {ucl_}", pval_])
                ytick.append(i)
            else:
                tval.append([' ', ' '])
                ytick.append(i)
        # set the x axis limits
        if max_value is None:
            maxi = self.df['UCL'].max() + self.df['UCL'].std()
            maxi = np.round(maxi, 1)
        else:
            maxi = max_value
        if min_value is None:
            mini = self.df['LCL'].min() - self.df['LCL'].std()
            mini = np.round(mini, 1)
        else:
            mini = min_value

        figsize = (14, 16)
        plt.figure(figsize=figsize)
        gspec = gridspec.GridSpec(1, 6)
        plot = plt.subplot(gspec[0, 0:4])
        tabl = plt.subplot(gspec[0, 4:])
        plot.set_ylim(-1, len(self.df))

        if self.scale == 'log':
            plot.set_xscale('log')

        plot.axvline(self.center, color=self.linec, zorder=1)
        plot.errorbar(x=self.df['OR2'],
                      y=self.df.index,
                      xerr=[self.df['LCL_dif'], self.df['UCL_dif']],
                      marker='None',
                      zorder=2,
                      ecolor=self.errc,
                      elinewidth=(size / size),
                      linewidth=0)
        plot.scatter(x=self.df['OR2'],
                     y=self.df.index,
                     c='orange', # self.pc,
                     s=(size * 25),
                     marker=self.shape,
                     zorder=3,
                     edgecolors='None')
        plot.vlines(x=1,
                    ymin=np.min(ytick),
                    ymax=np.max(ytick),
                    linestyle='--',
                    color='orange',
                    alpha=0.5)
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        plot.set_yticks(ytick)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi])
        plot.set_xticklabels([mini, self.center, maxi])
        plot.set_yticklabels(self.df['study'])
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()
        tb = tabl.table(cellText=tval,
                        cellLoc='center',
                        loc='right',
                        colLabels=[self.em, self.ci],
                        bbox=[0, t_adjuster, 1, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(text_size)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)
        plot.grid(axis='y',
                 alpha=0.7)
        plt.tight_layout()
        return plot

if __name__ == '__main__':
    presentation_path = 'C:/Users/giorg/OneDrive - Fundacion Raices Italo Colombianas/projects/miglab_bioserenity/results/presentation/reg_after_lasso'
    presentation_path = pathlib.Path(presentation_path)
    file_name = '0_PH2_Dentures_iter_reg.xlsx'
    pretty_table_path = presentation_path.joinpath(file_name)

    df_table = pd.read_excel(pretty_table_path)
    # unstack the OR and CI from single cell to multiple columns
    df_table[['OR', 'ci_low_bound', 'ci_high_bound']] = df_table['OR'].str.extract(r'([\d.]+)\s+\(([\d.]+),([\d.]+)\)')
    columns = ['variable', 'OR', 'ci_low_bound', 'ci_high_bound', 'p-value']
    df_table = df_table[columns]
    df_table[columns[1::]] = df_table[columns[1::]].astype(float)

    df_table.sort_values(by='p-value',
                         inplace=True)

    alpha_corr = 0.05 / df_table.shape[0]
    index_significance = np.abs(df_table['p-value'] - alpha_corr).idxmin()

    # df_table['color'] =  sns.color_palette('Paired')[7]
    # df_table['color'] = np.where(df_table['OR'] > 1, sns.color_palette('Paired')[7], sns.color_palette('Paired')[1])

    # let's show only the significant ones

    # TODO take this class and create one where we can also plot the p values
    forest_plot = EffectMeasurePlot(label=df_table.variable.tolist(),
                                    effect_measure=df_table.OR.tolist(),
                                    lcl=df_table.ci_low_bound.tolist(),
                                    ucl=df_table.ci_high_bound.tolist(),
                                    p_value=df_table['p-value'].to_list())

    forest_plot.plot(figsize=(16, 12), max_value=1.5)
    plt.tight_layout()
    plt.show()
