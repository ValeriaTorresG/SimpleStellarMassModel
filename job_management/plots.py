import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

def plot_correlation_matrix(r, cmap='mako'):
    filename = f'../data/rosette{r}.csv'
    column_names = [r'$M*$', r'$FLUX \thinspace G$', r'$FLUX \thinspace R$', r'$FLUX \thinspace Z$', r'$FLUX \thinspace W1$', r'$FLUX \thinspace W2$', r'$Z$']
    data = pd.read_csv(filename)
    data.drop('TARGET_ID', axis=1, inplace=True)
    corr = data.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(corr, cmap=cmap, square=True, annot=True, linewidths=0.5)
    ax.set_xticklabels(column_names, rotation=45, horizontalalignment='right')
    ax.set_yticklabels(column_names, rotation=0, horizontalalignment='right')
    ax.set_title(f'Correlation matrix - Rosette {r}', fontsize=11, pad=15, y=1)
    plt.tight_layout()
    fig.savefig(f'../plots/linear/corr_rosette{r}.png', dpi=360)

plot_correlation_matrix(r=3)