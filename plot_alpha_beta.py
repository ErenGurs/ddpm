import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
import numpy

def plot_beta(alpha_hat, beta):
    plt.plot(alpha_hat.cpu().numpy(), color='r', label='alpha_hat')
    plt.plot(torch.sqrt(alpha_hat).cpu().numpy(), color='g', label='sqrt(alpha_hat)')
    plt.plot(beta.cpu().numpy(), color='b', label='beta')
    plt.ylabel('alpha_hat/beta')
    plt.xlabel('t \in [0,1000]')
    plt.grid(True)

    plt.title("beta schedule, alpha_hat and sqrt(alpha_hat)")

    plt.savefig('beta_alpha_hat.png')


def plot_alpha_beta(alpha_hat, beta):
    # From example 
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    fig, ax1 = plt.subplots()

    #plt.plot(alpha_hat.cpu().numpy(), color='r', label='alpha_hat')
    color1 = 'tab:red'
    ax1.set_ylabel(r'$\beta_t$ schedule', color=color1)
    ax1.set_xlabel(r'$t \in [0,1000]$')
    #ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.1e+1'))
    ax1.plot(beta.cpu().numpy(), color=color1, label='$beta$ schedule')


    color2 = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis    
    ax2.set_ylabel(r'$\hat{\alpha}_t$', color=color2)
    #ax2.plot(torch.sqrt(alpha_hat).cpu().numpy(), color=color2, label='sqrt(alpha_hat)')
    ax2.plot(alpha_hat.cpu().numpy(), color=color2, label='alpha_hat')


    #plt.title("$\beta_t$ schedule and $\sqrt{\alpha_t}$")
    plt.grid(True)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()

    plt.savefig('eren_alpha_hat.png')