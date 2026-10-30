"""One look for every Realism plot: white background, the page's two colours (orange = ours now, blue = the real
system or what we will do), grey for references, thin axes."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OURS = '#c06000'          # the page's --now
REAL = '#1d62c2'          # the page's --new
REF = '0.45'
SEQ = plt.get_cmap('viridis')

plt.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 110, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5, 'legend.fontsize': 9, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': '0.9', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'figure.constrained_layout.use': True,
})


def dark():
    """Black background and white text (Luke, 29 Oct: results pages all black). Returns the colours to use instead
    of OURS, REAL, REF, the ink for single-series lines, and a sequential map that stays visible on black."""
    plt.rcParams.update({
        'figure.facecolor': 'black', 'axes.facecolor': 'black', 'savefig.facecolor': 'black',
        'text.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': '0.65', 'axes.titlecolor': 'white',
        'xtick.color': '0.85', 'ytick.color': '0.85', 'grid.color': '0.22', 'legend.labelcolor': 'white',
    })
    return '#f29a45', '#79aef5', '0.62', '0.92', (lambda v: SEQ(0.3 + 0.7 * v))


def save(fig, path):
    fig.savefig(path, facecolor=plt.rcParams['savefig.facecolor'])
    plt.close(fig)
    print('wrote', path)
