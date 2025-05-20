""""
Working with the following pipeline:
1. Lasso
2. Iterative Feature Regression
3. Xgboost

The pipeline is tested under different configurations, this script will test the results of each and evaluate which
is the best model
"""
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
from config.config import config
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from typing import Optional, Tuple, Union
import seaborn as sns
import ast
import itertools
from LassoRegBoostPipeline.model_analysis_plotter import ModelAnalysisPlotter

if __name__ == '__main__':
    path_results = config.get('models_path')
    # Load all JSON files into a DataFrame
    model_reports = []
    # List all output_report.json files in the directory
    for filepath in path_results.rglob('output_report.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
            data['model_file'] = filepath.name  # Keep track of the model file name
            model_reports.append(data)

    # Convert list of dictionaries to DataFrame
    df_reports = pd.DataFrame(model_reports)

    # Display the DataFrame to see the data
    print(df_reports)

    df_reports = df_reports.drop_duplicates(subset=['model_name'])

    # %%    PLOT THE F1 SCORES
    df_f1_score_melted = pd.melt(df_reports, id_vars=['model_name', 'split_gender'],
                        value_vars=['train_F1 Score', 'val_F1 Score', 'test_F1 Score'],
                                 var_name='F1 Score Type',
                        value_name='F1 Score')

    plt.figure(figsize=(12, 6))
    # Define color palette for F1 score types
    palette = {'train_F1 Score': 'lightblue', 'val_F1 Score': 'orange', 'test_F1 Score': 'lightgreen'}
    # Plot swarmplot
    sns.lineplot(data=df_f1_score_melted,
                 x='model_name',
                 y='F1 Score',
                 hue='F1 Score Type',
                 palette=palette,  # Set color palette
                 marker='o',  # Use markers for points
                 markersize=8)  # Set marker size
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')  # Rotate x-axis labels for better readability and center alignment
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
    plt.tight_layout()
    plt.grid(alpha=0.7)
    plt.show()
    # %%    PLOT THE FEATURE REDUCTION LENGTH
    plt.figure(figsize=(12, 6))
    df_lenfeat_melted = pd.melt(df_reports, id_vars=['model_name', 'split_gender'],
                        value_vars=['n_features_initial', 'n_features_afterlasso', 'n_features_afterreg'],
                        var_name='Reduced Features',
                        value_name='Number Features')
    palette = {'n_features_initial': 'lightblue', 'n_features_afterlasso': 'orange', 'n_features_afterreg': 'lightgreen'}
    # Plot swarmplot
    sns.lineplot(data=df_lenfeat_melted,
                 x='model_name',
                 y='Number Features',
                 hue='Reduced Features',
                 palette=palette,  # Set color palette
                 marker='o',  # Use markers for points
                 markersize=8)  # Set marker size
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')  # Rotate x-axis labels for better readability and center alignment
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
    plt.tight_layout()
    plt.grid(alpha=0.7)
    plt.show()
    # %% Confusion matrix of the models with best F1 Score

    best_train_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['train_F1 Score'].max(), :]
    plotter = ModelAnalysisPlotter(model_path=best_train_metric['model_path'].values[0])
    plotter.explore_model(sub_string='best train metrics')

    best_test_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['test_F1 Score'].max(), :]
    plotter = ModelAnalysisPlotter(model_path=best_test_metric['model_path'].values[0])
    plotter.plot_model_analysis(sub_string='best test metrics')


    from utils import EffectMeasurePlot
    def plot_model_analysis(model_path: str,
                            sub_strnig:Optional[str] = None,
                            figsize:Optional[Tuple]=(14, 12)):
        """
        Plot various analysis plots for a given model
        :param model_path: path to the model where images of the model results are present
        :param sub_strnig: str, to add to the title in case we want to be more clear on the model we are plotting
        :param figsize; tuple[int, int], figure size of the figure
        :return:
        """
        model_path = pathlib.Path(model_path)
        name_model = model_path.name.split('-')[3][2::]
        cm_test = model_path.joinpath('confusion_matrix_test.png')
        cm_train = model_path.joinpath('confusion_matrix_train.png')
        dist_target = model_path.joinpath('Distribution_Model.png')
        train_val_curve = model_path.joinpath('train_val_curve.png')

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
        if sub_strnig is not None:
            fig.suptitle(name_model+ f' {sub_strnig}')
        else:
            fig.suptitle(name_model)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    # train evaluation
    best_train_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['train_F1 Score'].max(), :]
    plot_model_analysis(model_path=best_train_metric['model_path'].values[0], sub_strnig='best train metrics')


    # test evaluation
    best_test_metric = df_reports.loc[df_reports['train_F1 Score'] == df_reports['test_F1 Score'].max(), :]
    plot_model_analysis(model_path=best_test_metric['model_path'].values[0], sub_strnig='best test metrics')

    def plot_regression_results(model_path: Union[str, pathlib.Path],
                            figsize:Optional[Tuple]=(14, 16)) :
        """
        From the GolbOutput.xlsx table we will plot the odds ratio of the last model used in the regression.
        Id the or plots already exists, it will display it, else, the plot will created and displayed from the
        global output excel file
        :param model_path: path to the model where the subfolders with the tables are located
        :param figsize: figure size to display the image
        :return:
        """
        if isinstance(model_path, str):
            model_path = pathlib.Path(model_path)

        output_path = model_path.joinpath('OlsFeatSelc','publish', 'or_plot.png')

        if output_path.exists():
            img = mpimg.imread(output_path)
            plt.figure(figsize=figsize)
            # Display the image
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        global_output_path = model_path.joinpath('OlsFeatSelc','iter', 'GlobOutput.xlsx')
        output_path = model_path.joinpath('OlsFeatSelc','publish', 'or_plot.png')

        df_global = pd.read_excel(global_output_path)
        # get the last model and the columns of interest
        df_last_model = df_global.loc[df_global['model'] == df_global.model.unique()[-1],
                                      ['variable', 'odds', 'odds_ci_low', 'odds_ci_high', 'P>|t|']]
        df_last_model = df_last_model.loc[df_last_model.variable != 'const', :]
        # make the OR plot
        forest_plot = EffectMeasurePlot(label=df_last_model.variable.tolist(),
                                        effect_measure=df_last_model.odds.tolist(),
                                        lcl=df_last_model.odds_ci_low.tolist(),
                                        ucl=df_last_model.odds_ci_high.tolist(),
                                        p_value=df_last_model['P>|t|'].tolist(),
                                        alpha=0.05)
        forest_plot.plot(figsize=figsize,
                         path_save=None, #output_path,
                         show=True)



    # %%    PLOT THE FEATURE REDUCTION MAP
    all_features_afterlasso = []
    for _, row in df_reports.iterrows():
        features = ast.literal_eval(row['features_afterlasso'])
        all_features_afterlasso.extend(features)

    # Count the occurrences of each feature
    feature_counts_afterlasso = pd.Series(all_features_afterlasso).value_counts()
    # Print the top 10 most significant features
    print("Top 10 most significant features after LASSO:")
    print(feature_counts_afterlasso.head(10))

    feature_counts_sorted = feature_counts_afterlasso.sort_values(ascending=False)
    feature_counts_sorted = feature_counts_sorted.to_frame().reset_index(drop=False)
    feature_counts_sorted.columns = ['features', 'counts']
    groups = feature_counts_sorted['counts'].unique()
    ncols = 2
    nrows = len(groups) // ncols + (len(groups) % ncols > 0)
    plt.figure(figsize=(16, 40))
    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    for n, group_ in enumerate(groups):
        ax = plt.subplot(nrows, ncols, n + 1)
        sns.barplot(data=feature_counts_sorted[feature_counts_sorted['counts'] == group_],
                    x='features',
                    y='counts',
                    ax=ax
                    )
        ax.set_title('Feature Counts After LASSO')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Count')
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()
    # %% plot heat map of the features
    df_heatmap_model_features = df_reports.loc[:, ['model_name', 'features_afterreg', 'test_F1 Score']].copy()
    df_heatmap_model_features.sort_values(by='test_F1 Score', inplace=True, ascending=False)
    df_heatmap_model_features.reset_index(inplace=True, drop=True)
    unique_features = []
    for idx, features in df_heatmap_model_features[['features_afterreg']].iterrows():
        unique_features.extend(features.apply(ast.literal_eval))

    unique_features = [item for sublist in unique_features for item in sublist]
    unique_features = sorted(list(set(unique_features)))
    df_heatmap_model_features[unique_features] = 0

    # populate the feature columns
    for idx, features in df_heatmap_model_features[['features_afterreg']].iterrows():
        features = features.apply(ast.literal_eval).iloc[0]
        for col in unique_features:
            if col in features:
                df_heatmap_model_features.loc[idx, col] = 1
    # plot the heatmap of features used in each model

    # plt.figure(figsize=(20, 12))
    # sns.heatmap(df_heatmap_model_features[unique_features],
    #             cmap='Blues',
    #             cbar=False,
    #             annot=False,
    #             # fmt='d',
    #             linewidths=.5)
    # plt.title('Feature Presence Heatmap')
    # plt.xlabel('Features')
    # plt.ylabel('Model Names')
    # plt.yticks(ticks=df_heatmap_model_features['model_name'].index, labels=df_heatmap_model_features['model_name'].values)
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()
    #
    # # Set the seaborn theme for better presentation
    # sns.set_theme(style="whitegrid")

    # Create the heatmap
    plt.figure(figsize=(20, 12))
    ax = sns.heatmap(df_heatmap_model_features[unique_features],
                     cmap='Blues',
                     cbar=False,
                     annot=False,
                     linewidths=.5)
    plt.title('Feature Presence Heatmap', fontsize=20)
    plt.xlabel('Features', fontsize=16)
    plt.ylabel('Model Names', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=90,
                       verticalalignment='center_baseline',  # 'top', 'bottom', 'center', 'baseline', 'center_baseline'
                       fontsize=12,
                       # ha='left'
                       )
    ax.set_yticks(ticks=df_heatmap_model_features['model_name'].index,
                  labels=df_heatmap_model_features['model_name'].values,
                  fontsize=12,
                  ha='right', # 'center', 'right', 'left'
                  verticalalignment='center',
                  rotation=0
                  )
    # Create a secondary y-axis
    ax2 = ax.twinx()
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(df_heatmap_model_features['test_F1 Score'].values[::-1],
                        rotation=0,
                        verticalalignment='center',
                        fontsize=12)
    ax2.set_ylabel(ylabel='F1 Score')
    plt.tight_layout()
    plt.grid(False)
    plt.show()


    # %%    PLOT THE LASSO COEFFICIENTS
    def plot_lasso_coefficients(row):
        coefficients_dict = eval(row)
        classes = list(coefficients_dict.keys())
        features = list(coefficients_dict[classes[0]].keys())  # Assuming all classes have the same features

        for class_name in classes:
            class_coefficients = [coefficients_dict[class_name][feature] for feature in features]
            plt.bar(features, class_coefficients, label=class_name)

        plt.xlabel('Features')
        plt.ylabel('Lasso Coefficients')
        plt.title('Lasso Coefficients for Each Class')
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    # Apply the function to each row in the DataFrame
    df_reports['lasso_coefficients'].apply(plot_lasso_coefficients)

    # %%
    # join the Lasso classifier coefficients in a single dataframe
    indexes = [f'class_{i}' for i in range(0, 4)]
    features = ast.literal_eval(df_reports.loc[0, 'features_initial'])
    indexes = list(itertools.chain.from_iterable(itertools.repeat(indexes, df_reports.shape[0])))
    repeated_model_names = list(
        itertools.chain.from_iterable(itertools.repeat(name, 4) for name in df_reports['model_name']))

    coefficients_df = pd.DataFrame(np.nan,
                                   columns=features,
                                   index=indexes)
    coefficients_df['model'] = repeated_model_names

    # Iterate over rows in the original DataFrame
    for index, row in df_reports.iterrows():
        lasso_dict = eval(row['lasso_coefficients'])
        lasso_df = pd.DataFrame(lasso_dict).T
        lasso_df['model'] = row.model_name
        coefficients_df.loc[coefficients_df['model'] == row.model_name, lasso_df.columns] = lasso_df

    # coefficients_df.replace(0, np.nan, inplace=True)
    coefficients_df.reset_index(inplace=True, drop=False, names='class')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=coefficients_df,
                x='class',
                y='EpReading',
                hue='model')
    plt.xlabel('Class')
    plt.ylabel('Lasso Coefficients')
    plt.title('Lasso Coefficients for EpReading Across Classes (Grouped Bar Plot)')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    # Option 2: Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=coefficients_df,
                x='class',
                y='EpReading')
    plt.xlabel('Class')
    plt.ylabel('Lasso Coefficients')
    plt.title('Distribution of Lasso Coefficients for EpReading Across Classes (Box Plot)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Option 3: Heatmap
    plt.figure(figsize=(12, 6))
    heatmap_data = coefficients_df.pivot_table(index='class',
                                               columns='model',
                                               values='EpReading')
    ax = sns.heatmap(heatmap_data,
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                linewidths=.5)
    plt.xlabel('Model')
    plt.ylabel('Class')
    plt.title('Lasso Coefficients Heatmap for EpReading')
    plt.xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Option 4: Line Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=coefficients_df, x='class', y='EpReading', hue='model', marker='o')
    plt.xlabel('Class')
    plt.ylabel('Lasso Coefficients')
    plt.title('Lasso Coefficients Trend for EpReading Across Classes (Line Plot)')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    # average the coefficients across the class
    feature_columns = coefficients_df.columns[1:-1]
    average_coefficients = coefficients_df.groupby('class')[feature_columns].mean()

    # Option 1: Scatter Plot
    data_to_plot = average_coefficients.T  # Transpose to get features on x-axis and classes as lines
    features = data_to_plot.index
    classes = data_to_plot.columns
    plt.figure(figsize=(15, 8))
    for class_label in classes:
        plt.scatter(x=features,
                    y=data_to_plot[class_label], label=class_label, marker='o')
    plt.title('Average Lasso Coefficients per Class')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Magnitude')
    plt.xticks(rotation=90)
    plt.legend(title='Class')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # average the coefficients within each model and plot a heatmatp
    average_coefficients = coefficients_df.groupby('model')[feature_columns].mean()
    plt.figure(figsize=(70,12 ))
    ax = sns.heatmap(average_coefficients,
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                linewidths=.5,
                     annot_kws={"size": 8})
    plt.xlabel('Features After Lasso')
    plt.ylabel('Models')
    plt.title('Lasso Coefficients All Models All Features')
    plt.xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Option 2: Bar Plot
    sampled_features = features[:20]  # Using only the first 20 features for simplicity

    n_sampled_features = len(sampled_features)
    ncols = 1  # Number of columns in subplot
    nrows = n_sampled_features // ncols + (n_sampled_features % ncols > 0)  # Number of rows

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows * 4), squeeze=False)
    fig.suptitle('Average Lasso Coefficients for Each Class by Feature (Sampled)', fontsize=16)
    width = 0.2
    # Flatten the array of axes for easy iteration
    axs = axs.flatten()
    for idx, feature in enumerate(sampled_features):
        # Locations for the groups on x-axis
        x = np.arange(len(classes))

        # Plotting
        axs[idx].bar(x - width, data_to_plot.loc[feature, classes[0]], width, label=classes[0])
        axs[idx].bar(x, data_to_plot.loc[feature, classes[1]], width, label=classes[1])
        axs[idx].bar(x + width, data_to_plot.loc[feature, classes[2]], width, label=classes[2])
        axs[idx].bar(x + 2 * width, data_to_plot.loc[feature, classes[3]], width, label=classes[3])

        axs[idx].set_title(feature)
        axs[idx].set_xticks(x + width / 2)
        axs[idx].set_xticklabels(classes)
        axs[idx].axhline(0, color='grey', linewidth=0.8)
        axs[idx].legend()

    # Remove empty subplots
    for ax in axs[n_sampled_features:]:
        ax.remove()

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # %% Evaluate best model
    best_model = df_reports.loc[df_reports['test_F1 Score'] == df_reports['test_F1 Score'].max(), :]
    # get confusion matrix
    img_path = pathlib.Path(best_model['model_path'].values[0]).joinpath('confusion_matrix_test.png')
    img = mpimg.imread(img_path)
    plt.figure(figsize=(20, 16))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    img_path = pathlib.Path(best_model['model_path'].values[0]).joinpath('train_val_curve.png')
    img = mpimg.imread(img_path)
    plt.figure(figsize=(20, 16))
    plt.imshow(img)
    plt.axis('off')
    plt.show()











