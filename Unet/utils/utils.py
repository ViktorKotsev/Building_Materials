import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_uncertainty(uncertainty):
    # Ensure the uncertainty has three channels for the three classes
    if uncertainty.shape[0] == 3:
        # Calculate the mean uncertainty across classes
        mean_uncertainty = np.mean(uncertainty, axis=0)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        class_titles = ['Class 1 Uncertainty', 'Class 2 Uncertainty', 'Class 3 Uncertainty', 'Mean Uncertainty']
        
        for i in range(3):
            ax = axes[i]
            cax = ax.imshow(uncertainty[i], cmap='hot', interpolation='nearest')
            fig.colorbar(cax, ax=ax)
            ax.set_title(class_titles[i])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
        
        # Plot the mean uncertainty map
        ax = axes[3]
        cax = ax.imshow(mean_uncertainty, cmap='hot', interpolation='nearest')
        fig.colorbar(cax, ax=ax)
        ax.set_title(class_titles[3])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        
        # Save the plot to a file
        plt.savefig('uncertainty_plot.png')