"""Static report charts and interactive dashboard charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import FEATURES

def radar_chart(summary):
    columns = ['Study_Hours_Per_Day', 'Attendance_Percentage', 'Assignment_Completion_Percentage', 'Average_Test_Score', 'Class_Interactions']
    maxima = [10, 100, 100, 100, 50]
    figure = go.Figure()
    for _, row in summary.iterrows():
        values = [row[c] / limit * 100 for c, limit in zip(columns, maxima)]
        labels = [c.replace('_', ' ') for c in columns]
        figure.add_trace(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill='toself', name=row.Learning_Behaviour))
    figure.update_layout(title='Cluster characteristics (% of configured maximum)', polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return figure

def pca_chart(coordinates, centers=None, three_dimensional=False):
    labels = {'PC1': 'PCA Component 1', 'PC2': 'PCA Component 2', 'PC3': 'PCA Component 3'}
    if three_dimensional:
        return px.scatter_3d(coordinates, x='PC1', y='PC2', z='PC3', color='Learning_Behaviour', labels=labels, title='Student similarity in three PCA dimensions', opacity=.65)
    figure = px.scatter(coordinates, x='PC1', y='PC2', color='Learning_Behaviour', labels=labels, title='Student learning behaviour clusters', opacity=.6)
    if centers is not None:
        figure.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers', marker=dict(symbol='x', size=16, color='black'), name='Cluster centers'))
    return figure

def save_charts(raw, summary, evaluation, coordinates, centers, root):
    destination = root / 'outputs/charts'
    plt.style.use('seaborn-v0_8-whitegrid')
    def save(name):
        plt.tight_layout()
        plt.savefig(destination / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    for column in ['Study_Hours_Per_Day', 'Attendance_Percentage', 'Average_Test_Score']:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(raw[column].dropna(), bins=30, color='#0d9488', edgecolor='white')
        ax.set(xlabel=column.replace('_', ' '), ylabel='Students', title=column.replace('_', ' ') + ' distribution')
        save('hist_' + column)
    for column in ['Study_Hours_Per_Day', 'Attendance_Percentage', 'Assignment_Completion_Percentage']:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(raw[column], raw.Average_Test_Score, s=9, alpha=.3, color='#2563eb')
        ax.set(xlabel=column.replace('_', ' '), ylabel='Average test score', title=column.replace('_', ' ') + ' vs test score')
        save('scatter_' + column)
    fig, ax = plt.subplots(figsize=(11, 9))
    matrix = raw[FEATURES].corr()
    heat = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(FEATURES)), [c.replace('_', ' ') for c in FEATURES], rotation=90)
    ax.set_yticks(range(len(FEATURES)), [c.replace('_', ' ') for c in FEATURES])
    ax.set_title('Pearson correlation of learning habits')
    fig.colorbar(heat, ax=ax)
    save('correlation_heatmap')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.flat, FEATURES[:6]):
        ax.boxplot(raw[column].dropna())
        ax.set(title=column.replace('_', ' '), ylabel='Observed value', xticks=[])
    save('feature_boxplots')
    for metric, title in [('Inertia', 'Elbow method'), ('Silhouette', 'Silhouette score by K')]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(evaluation.K, evaluation[metric], 'o-', color='#0d9488')
        ax.set(xlabel='Number of clusters (K)', ylabel=metric, title=title, xticks=evaluation.K)
        save(metric.lower())
    for column in ['Student_Count', 'Study_Hours_Per_Day', 'Attendance_Percentage', 'Assignment_Completion_Percentage', 'Average_Test_Score', 'Previous_Semester_Marks']:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(summary.Learning_Behaviour, summary[column], color=plt.cm.Set2(np.linspace(0, 1, len(summary))))
        ax.set(xlabel='Learning behaviour', ylabel=column.replace('_', ' '), title=column.replace('_', ' ') + ' by cluster')
        ax.tick_params(axis='x', rotation=20)
        save('cluster_' + column)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, group in coordinates.groupby('Learning_Behaviour'):
        ax.scatter(group.PC1, group.PC2, s=10, alpha=.45, label=name)
    ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=150, color='black', label='Cluster centers')
    ax.set(xlabel='PCA Component 1', ylabel='PCA Component 2', title='Student learning behaviour clusters')
    ax.legend(fontsize=8)
    save('pca_2d')
    # Interactive HTML is self-contained and opens without an internet connection.
    radar_chart(summary).write_html(destination / 'radar.html', include_plotlyjs=True)
    pca_chart(coordinates, centers, True).write_html(destination / 'pca_3d.html', include_plotlyjs=True)
