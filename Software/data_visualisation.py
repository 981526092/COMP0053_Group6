from matplotlib import pyplot as plt
import numpy as np

# Plot the target class distribution in a pie chart
def plot_target(y):
    plt.pie(np.unique(y,return_counts=True)[1],labels=['0: Not-Protective','1: Protective'],autopct='%1.1f%%')
