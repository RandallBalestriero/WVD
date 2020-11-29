from wvd import banks
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib
from scipy.signal import hilbert

cmap = matplotlib.cm.get_cmap("viridis")


def plot_gauss(X, Y, gaussians, name):
    for gauss in gaussians:
        plt.contour(X, Y, gauss, [0.05], linewidths=0.3)

    plt.savefig(name + "_contours.png")
    plt.close()

    for k in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        for gauss in gaussians:
            plt.contour(X, Y, np.roll(gauss, 2 * k, 1), [0.05], linewidths=0.3)

    plt.savefig(name + "_translations.png")
    plt.close()


def fancy_plot(filters, name, gaussians=None, limit=False):
    w_filters = np.fft.fft(np.fft.fftshift(filters, axes=-1)).real
    filters /= np.abs(filters).max(1, keepdims=True)
    w_filters /= np.abs(w_filters).max(1, keepdims=True)
    w_filters = w_filters[:, : len(w_filters[0]) // 2]

    fig = plt.figure(
        constrained_layout=True, figsize=(11 / (1 + int(limit)), 4.5)
    )
    gs = GridSpec(5, 5, figure=fig)

    axtop = fig.add_subplot(gs[:2, 1:])
    axleft = fig.add_subplot(gs[2:, :1])
    axbottom = fig.add_subplot(gs[2:, 1:])

    colors = cmap(np.linspace(0, 0.85, len(filters)))

    X, Y = np.meshgrid(
        np.arange(len(filters[0])), np.arange(len(w_filters[0]))
    )

    for k, (color, left, right) in enumerate(zip(colors, w_filters, filters)):
        if gaussians is None:
            Z = left[:, None] * np.abs(right)
        else:
            x, y = np.meshgrid(
                np.linspace(-6, 6, len(filters[0])),
                np.linspace(0, np.pi, len(filters[0]) // 2),
            )
            pos = np.stack([x.reshape(-1), y.reshape(-1)], 1)
            Z = gaussians[k].pdf(pos).reshape(x.shape)
        Z /= Z.max()
        cs = axbottom.contourf(
            X, Y, Z, [0.02, 2], alpha=0.3, linewidths=1, colors=[color],
        )
        axbottom.contour(
            cs, linestyles="dashed", colors="k", linewidths=0.3, alpha=0.6
        )

    for k, (color, left) in enumerate(zip(colors, w_filters)):
        axleft.plot(
            left + 1.4 * k, np.arange(len(left)), color=color, alpha=0.6,
        )
        axleft.plot(
            np.zeros(len(left)) + 1.4 * k,
            np.arange(len(left)),
            color="black",
            linewidth=0.3,
        )
    axleft.set_ylim([0, len(w_filters[0])])
    axleft.invert_xaxis()

    for k, (color, right) in enumerate(zip(colors, filters)):
        if np.iscomplex(right).any():
            axtop.plot(right.real + 1.4 * k, color=color, alpha=0.6)
            axtop.plot(
                right.imag + 1.4 * k,
                color=color,
                alpha=0.6,
                linestyle="dashed",
            )
        else:
            axtop.plot(right + 1.4 * k, color=color, alpha=0.6)
        axtop.plot(
            np.zeros(len(right)) + 1.4 * k, color="black", linewidth=0.3,
        )
    axtop.set_xlim([0, len(filters[0])])

    for i, ax in enumerate(fig.axes):
        ax.set_xticks([])
        ax.set_yticks([])

    if limit:
        axtop.set_xlim(
            [len(filters[0]) // 4, 3 * len(filters[0]) // 4,]
        )
        axbottom.set_xlim(
            [len(filters[0]) // 4, 3 * len(filters[0]) // 4,]
        )

    positions = np.arange(len(filters)) * 1.4
    axleft.set_xticks(positions, [""] * len(positions))
    # axleft.tick_params("x", length=50, width=100, which="major")
    axleft.tick_params("x", length=3, width=2, which="minor")

    axtop.set_yticks(positions, [""] * len(positions))
    axtop.tick_params("y", length=3, width=2, which="minor")

    axtop.set_xlabel("Time", fontsize=23)
    axbottom.set_ylabel("Frequency", fontsize=23)

    plt.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.985,
        bottom=0.015,
        wspace=0.22 + 0.22 * int(limit),
        hspace=0.55,
    )

    plt.savefig(name + "_fancy.png")
    plt.close()


def plot_usual_banks(J, Q):
    N = 15 * 2 ** J
    morlets = banks.morlet(J, Q).get()
    fancy_plot(morlets, "morlets")
    sincs = banks.sinc(J, Q, N).get()
    fancy_plot(sincs, "sincs")
    stft_small = banks.stft(0.01, J, Q // 2, 1200).get()
    fancy_plot(stft_small, "stft_small", limit=True)
    stft_large = banks.stft(0.1, J, Q * 2, 550).get()
    fancy_plot(stft_large, "stft_large", limit=True)
    chirps, gaussians = banks.chirp(J, Q)
    chirps = chirps.get()
    fancy_plot(chirps, "chirps", gaussians)


def plot_2d_banks(J, Q):
    stft_small = banks.gaussian2d(1024, J, Q, init="stft", window=256).get()
    stft_large = banks.gaussian2d(1024, J, Q, init="stft", window=1024).get()
    gaussians = banks.gaussian2d(1024, J, Q).get()

    X, Y = np.meshgrid(
        np.arange(gaussians.shape[2]), np.arange(gaussians.shape[1])
    )

    plot_gauss(X, Y, gaussians, "gabor")
    plot_gauss(X, Y, stft_small, "stft_small")
    plot_gauss(X, Y, stft_large, "stft_large")


if __name__ == "__main__":
    # plot_usual_banks(6, 2)
    plot_2d_banks(6, 2)
