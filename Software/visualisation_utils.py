from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Plot the target class distribution in a pie chart
def plot_target(y):
    plt.pie(np.unique(y,return_counts=True)[1],labels=['0: Not-Protective','1: Protective'],autopct='%1.1f%%')


def plot_confusion_matrix(confusion_matrix):
    # Set the seaborn style and font size
    sns.set_style("whitegrid")
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5, cbar=False)
    # Set labels
    ax.set_ylabel('True label', fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_title('Confusion Matrix Heatmap', fontsize=16)
    ax.set_xticklabels(['0', '1'], fontsize=12)
    ax.set_yticklabels(['0', '1'], fontsize=12)
    plt.show()


def plot_model_scores(model_scores):
    # Convert
    df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Accuracy', 'F1 Score', 'Recall', 'Precision'])
    
    # Reset
    df = df.reset_index().rename(columns={'index': 'Model'})
    df = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')


    sns.set_style("whitegrid")
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot the grouped bar chart
    ax = sns.barplot(data=df, x='Model', y='Score', hue='Metric', palette='bright', alpha=0.8)

    # Set the axis labels and title
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    plt.title('Model Performance Metrics')
    plt.xticks(rotation=45, ha='right')

    # Get the maximum score for each metric
    max_scores = df.groupby('Metric')['Score'].transform(max)

    # Add a star marker
    for i, metric in enumerate(df['Metric'].unique()):
        max_score = max_scores[df['Metric'] == metric].max()
        model = df.loc[(df['Metric'] == metric) & (df['Score'] == max_score), 'Model'].iloc[0]
        x = df.loc[df['Model'] == model].index[0] + i * 0.25 - 0.375  # Adjust the x position to the center of the bar
        y = max_score
        ax.plot([x], [y], marker='*', markersize=15, markeredgewidth=2, markeredgecolor='black', zorder=10)
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(x, y+0.02), fontsize=12, va='center', ha='center', fontweight='bold')


    plt.show()


