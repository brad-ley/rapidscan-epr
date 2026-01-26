import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

def plot_tukey_windows():
    N = 1000
    alphas = [0, 0.25, 0.5, 0.75, 1]
    
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        window = tukey(N, alpha=alpha)
        plt.plot(window, label=f'alpha = {alpha}')
        
    plt.title('Tukey Window shapes for varying alpha values')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('tukey_windows.png')
    # print("Plot saved as tukey_windows.png")
    
    # Show the plot if interactive
    plt.show()

if __name__ == "__main__":
    plot_tukey_windows()
