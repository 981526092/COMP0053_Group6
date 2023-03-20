from matplotlib import pyplot as plt
import numpy as np
def plot_target(y_train):
    plt.pie(np.unique(y_train[:,0],return_counts=True)[1],labels=['1: Protective','0: Not-Protective'],autopct='%1.1f%%')