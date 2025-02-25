import os.path
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt, patches

import plotting
from bands import BandGetter
from chain_model import ChainModel
from harper import Harper
from localization_analyzer import LocalizationAnalyzer
from localization_measure import LocalizationMeasure
from params_range import ParamsRange
from plotting_utils import get_label, flatten_axes
from utils import logger, result_logger

CALCULATIONS_ONLY = os.getenv("BUILD_PLOTS_CALCULATIONS_ONLY", "0") == "1"
DO_COMPUTE_BANDS = os.getenv("BUILD_PLOTS_DO_COMPUTE_BANDS", "0") == "1"
PLOT_PGF = True


DO_PLOT = {
    "harper": False,
    "ssh": False,
    "sshh": False,
    "localization_harper": False,
    "localization_ssh": False,
    "localization_sshh": False,
    "gsd_sshh": False,
    "edgestates_harper": False,
    "edgestates_sshh": True,
}


ROOT = str(Path(__file__).parent.parent.joinpath("tex").joinpath("plots"))
DIRS: dict[str, str] = {key: os.path.join(ROOT, key) for key in ("png", "pgf")}

L_PHI_HOP = r'\varphi_\mathrm{hop}/\frac{\pi}{4}'
L_PHI_H = r'\varphi_\mathrm{H}/\frac{\pi}{4}'

BOX_EDGECOLOR = "red"
BOX_LINEWIDTH = 1

class Figsize:
    HUGE = (20/3, 5)
    BIG = (5.6, 4.2)
    NORMAL = (5, 3.75)

    BIG_FLAT = (5.6, 2.2)
    HUGE_QUADRATIC = HUGE[0], HUGE[0]

    HALF_NORMAL = 2.6, 2.6 / 4 * 3
    HALF_SMALL = 2, 1.5

    TWO_THIRDS_NORMAL = 3.4, 2.55
    ONE_THIRD_NORMAL = 1.7, 2.55


def savefig(filename, figure: plt.Figure | None = None):
    if not CALCULATIONS_ONLY:
        figure = figure or plt.gcf()
        logger.info("Saving png %s", filename)
        figure.savefig(os.path.join(DIRS["png"], filename + ".png"), bbox_inches='tight', dpi=300)
        if PLOT_PGF:
            logger.info("Saving pgf %s", filename)
            pgf_filename = os.path.join(DIRS["pgf"], filename + ".pgf")
            figure.savefig(pgf_filename, bbox_inches='tight')
            with open(pgf_filename, "r", encoding="utf-8") as f:
                pgf = f.read()
            png_filename = lambda i: f"{filename}-img{i}.png"
            img = 0
            while png_filename(img) in pgf:
                pgf = pgf.replace(png_filename(img), os.path.join("plots", "pgf", png_filename(img)))
                img += 1
            if img:
                with open(pgf_filename, "w", encoding="utf-8") as f:
                    f.write(pgf)
    plt.close(figure)


if __name__ == '__main__':

    import sys
    try:
        plot_all = sys.argv[1] == "1"
    except IndexError:
        plot_all = False

    do_plot = {key: value or plot_all for key, value in DO_PLOT.items()}

    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

    F = Harper.IrrationalFrequency
    FS = Figsize

    beta: float
    states: Sequence[plotting.State]
    alphas: Sequence[Harper.Frequency]
    alphas_special: Sequence[Harper.Frequency]


## Harper model

    if do_plot["harper"]:

        _, ax = plt.subplots(figsize=FS.NORMAL)
        d_alpha = .001
        N = 200
        alpha = .8
        ms = .4
        params = ParamsRange("alpha", (d_alpha, 1 - d_alpha, d_alpha), phi_H=1, N=N)
        plotting.plot_eigvals(params=params, ax=ax, text="", do_compute_bands=False,
                              xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                              line_style=" ", marker="o", markersize=ms, markerfacecolor="black", markeredgewidth=0,
                              alpha=alpha, ylim=(-2, 2))
        ax.set_xlim(0, 1)
        savefig("harper_butterfly")

        factor: float | None

        for factor, label in ((None, "commensurable"), (F.GOLDEN_NUMBER, "incommensurable")):

            d_alpha = .001
            _, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)

            if factor is None:
                params = ParamsRange("alpha", (d_alpha, 1-d_alpha, d_alpha), phi_H=1.2, N=N)
            elif isinstance(factor, Harper.IrrationalFrequency):
                params = ParamsRange("alpha", (d_alpha, d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=1.2, N=N)
            else:
                raise ValueError(factor)

            factor = factor or 1
            states = ((params[int(599/factor)], 80), (params[int(599/factor)], 60),
                      (params[int(609/factor)], 80), (params[int(609/factor)], 60))

            plotting.plot_eigvals(params=params, ax=ax, text="", do_compute_bands=False,
                                  xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                                  line_style=" ", marker="o", markersize=ms, markerfacecolor="black", markeredgewidth=0,
                                  states=states, alpha=alpha, ylim=(-2, 2))
            ax.set_xlim(0, 1)
            savefig(f"harper_butterfly_{label}")

            _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
            for i, state in enumerate(states):
                ax = flatten_axes(axs)[i]
                plotting.plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]),
                                    color_index=i, show_xlabel=i == 3, show_ylabel=True)
                alpha = state[0].harper_not_none.alpha
                alpha_str = f"{alpha/factor:.3f}" + r"\alpha_\mathrm{G}$" + "\n$=" + str(alpha)[:6] + r"\ldots" if label.startswith("in") else str(alpha)
                ax.text(.96, .9, r"$\alpha=" + alpha_str + r"$", ha="right", va="top", fontsize="x-small",
                        transform=ax.transAxes)
            savefig(f"harper_butterfly_states_{label}")


        fig, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
        params = ParamsRange("phi_H", (0, 2, .01), N=50, alpha=F.GOLDEN_NUMBER)
        states = (params[80], 23), (params[120], 23), (params[80], 10), (params[120], 10)
        plotting.plot_eigvals(params=params, ax=ax, text="", states=states, do_compute_bands=DO_COMPUTE_BANDS)
        savefig("harper_spectrum_phi_H")
        _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
        for i, state in enumerate(states):
            ax = flatten_axes(axs)[i]
            plotting.plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]),
                                color_index=i, show_xlabel=i == 3, show_ylabel=True)
            ax.text(.04, .9, f"${L_PHI_H}={state[0].harper_not_none.phi_H:.1f}$",
                    ha="left", va="top", fontsize="x-small", transform=ax.transAxes)
        savefig("harper_states_phi_H")


        _, axs = plt.subplots(2, 2, figsize=FS.BIG, sharex=True, sharey=True)
        for i, N_or_None in enumerate((20, 50, 200, None)):

            params = ParamsRange("phi_H", (0, 2, .01), N=(N_or_None or 10),
                                 alpha=F.GOLDEN_NUMBER)
            plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=f"$N={N_or_None}$" if N_or_None else "",
                                  ylim=(None, None),
                                  xlabel=r"$\varphi_H / \frac{\pi}{4}$"*(i>1),
                                  ylabel="$E$"*((i+1)%2),
                                  do_compute_bands=DO_COMPUTE_BANDS,
                                  plot_spectrum=bool(N_or_None))
        savefig("harper_spectrum_N")


        _, axs = plt.subplots(2, 2, figsize=FS.BIG, sharex=True, sharey=True)
        for i, alpha in enumerate((F.GOLDEN_NUMBER, F.GOLDEN_NUMBER_REDUCED, F.SEC_MOST_IRRAT, Harper.Frequency(.65))):
            params = ParamsRange("phi_H", (0, 2, 0.01), N=50, alpha=alpha)
            plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=f"$\\alpha={get_label(alpha).replace('$', '')}$", ylim=(-1.7, 1.7),
                                  xlabel=r"$\varphi_H / \frac{\pi}{4}$"*(i>1),
                                  ylabel="$E$"*((i+1)%2), do_compute_bands=DO_COMPUTE_BANDS)
        savefig("harper_spectrum_alpha")


        phi_Hs = {"metallic": .8, "insulator": 1.2}
        for label, phi_H in phi_Hs.items():

            params = ParamsRange("beta", (0, 2 * np.pi, np.pi / 100),
                                 N=50, phi_H=phi_H, alpha=F.GOLDEN_NUMBER)
            p = params[140]
            assert p.harper.beta == 1.4 * np.pi, p.harper.beta / np.pi
            states = (p, 43), (p, 21), (p, 20), (p, 12)
            _, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
            plotting.plot_eigvals(params=params, ax=ax, text="",
                                  xaxis=plotting.xaxis_getter("harper.beta", np.pi), xlabel=r"$\beta / \pi$",
                                  states=states, do_compute_bands=DO_COMPUTE_BANDS)
            zoom = (0.81, .86, 1.21, 1.23) if label == "metallic" else (.95243, .95244, -.941944, -.941934) if label == "insulator" else None
            if zoom:
                rect = patches.Rectangle(
                    (0.02, -1.4), (.8 if label == "metallic" else .7), 1.1,
                    facecolor='white', alpha=.6, edgecolor='none', zorder=10
                )
                ax.add_patch(rect)
                ax.plot([.5*(zoom[0]+zoom[1])], [.5*(zoom[2]+zoom[3])], 's', markerfacecolor='none',
                        markersize=10, markeredgewidth=BOX_LINEWIDTH, markeredgecolor=BOX_EDGECOLOR)
                ax.add_patch(rect)
                inset_ax = ax.inset_axes((.12, -1.3, .4, .7), zorder=20, transform=ax.transData)
                inset_ax.xaxis.tick_top()
                inset_ax.yaxis.tick_right()
                inset_ax.set_xticks(zoom[:2])
                inset_ax.set_yticks(zoom[2:])
                if label == "insulator":
                    inset_ax.set_xticklabels((r"$x_0$", r"$x_1$"))
                    inset_ax.set_yticklabels((r"$y_0$", r"$y_1$"))
                    txt = r"$x_0 = 0.95243\pi$", r"$x_1=0.95244\pi$", r"$y_0=-0.941944$", r"$y_1=-0.941934$"
                    ax.text(.75, .5, "\n".join(txt), ha="left", va="center", fontsize="x-small")
                for ticklabel in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                    ticklabel.set_fontsize("x-small")
                precision = .000001 if label == "insulator" else .001
                inset_params = ParamsRange("beta", (np.pi*zoom[0], np.pi*zoom[1], np.pi*precision),
                                     N=50, phi_H=phi_H, alpha=F.GOLDEN_NUMBER)
                plotting.plot_eigvals(params=inset_params, ax=inset_ax, text="",
                                      xaxis=plotting.xaxis_getter("harper.beta", np.pi), xlabel=r"",
                                      ylabel="", do_compute_bands=DO_COMPUTE_BANDS)
                inset_ax.set_ylim(*zoom[2:])
            savefig(f"harper_spectrum_beta_{label}")
            _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
            for i, state, in enumerate(states):
                plotting.plot_state(state, plot_probability=True, ax=flatten_axes(axs)[i], psi_index=str(state[1]),
                                    color_index=i, show_xlabel=i == 3, show_ylabel=True)
            savefig(f"harper_states_beta_{label}")


## SSH model

    if do_plot["ssh"]:

        fig, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
        params = ParamsRange("phi_hop", (0, 2, .01), N=50)
        states = (params[80], 25), (params[120], 25), (params[80], 12), (params[120], 12)
        plotting.plot_eigvals(params=params, ax=ax, text="", states=states,
                              xlabel=r"$\varphi_\mathrm{hop} / \frac{\pi}{4}$",
                              xaxis=plotting.xaxis_getter("phi_hop"), do_compute_bands=True)
        savefig("ssh_spectrum_phi_hop")
        _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
        for i, state in enumerate(states):
            ax = flatten_axes(axs)[i]
            plotting.plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]), color_index=i,
                                show_xlabel=i == 3, show_ylabel=True, show_legend=False)
            ax.text(.96, .9, f"${L_PHI_HOP}={state[0].phi_hop:.1f}$",
                    ha="right", va="top", fontsize="x-small", transform=ax.transAxes)
        savefig("ssh_states_phi_hop")


        _, axs = plt.subplots(2, 2, figsize=FS.BIG, sharex=True, sharey=True)
        for i, N in enumerate((20, 50, 100, 200)):
            params = ParamsRange("phi_hop", (0, 2, .01), N=N)
            plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=f"$N={N}$", ylim=(None, None),
                                  xaxis=plotting.xaxis_getter("phi_hop"),
                                  xlabel=r"$\varphi_\mathrm{hop} / \frac{\pi}{4}$"*(i>1),
                                  ylabel="$E$"*((i+1)%2), do_compute_bands=True)
        savefig("ssh_spectrum_N")


## SSHH model

    if do_plot["sshh"]:

        PHI_HOP = (.75, "trivial", (1.01, 1.04, .23, .26)), (1.25, "topological", (.8, 1, -.1, .1))
        PHI_H = (.25, "metallic", None), (1.25, "insulator", (1.6, 1.62, -.03, -.02))
        ALPHA = F.GOLDEN_NUMBER_REDUCED

        N = 200
        d_alpha = .001
        alpha = .8
        ms = .4

        f0, axs0 = plt.subplots(3, 3, figsize=FS.HUGE_QUADRATIC, sharex=True, sharey=True,
                              gridspec_kw={})
        plt.subplots_adjust(right=0.8)

        for factor, label in ((None, "commensurable"), (F.GOLDEN_NUMBER, "incommensurable")):


            _, axs = plt.subplots(3, 3, figsize=FS.HUGE_QUADRATIC, sharex=True, sharey=True,
                                  gridspec_kw={})
            plt.subplots_adjust(right=0.8)
            i = 0
            for phi_hop in (.7, 1.3, 1.6):
                for phi_H in (.2, .5, 1):
                    ax = flatten_axes(axs)[i]
                    ax0 = flatten_axes(axs0)[i]

                    if factor is None:
                        params = ParamsRange("alpha", (d_alpha, 1 - d_alpha, d_alpha), phi_H=phi_H, phi_hop=phi_hop, N=N)
                        params0 = ParamsRange("alpha", (d_alpha, .5 - d_alpha, d_alpha), phi_H=phi_H, phi_hop=phi_hop, N=N)
                    elif isinstance(factor, Harper.IrrationalFrequency):
                        params = ParamsRange("alpha", (d_alpha, d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=phi_H, phi_hop=phi_hop, N=N)
                        params0 = ParamsRange("alpha", (d_alpha * (int(1/d_alpha/factor)//2 + 1), d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=phi_H, phi_hop=phi_hop, N=N)
                    else:
                        raise ValueError(factor)

                    for a, p in ((ax, params), (ax0, params0)):
                        plotting.plot_eigvals(params=p, ax=a, do_compute_bands=False, text="",
                                              xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                                              line_style=" ", marker="o", markersize=ms, markerfacecolor="black",
                                              markeredgewidth=0,
                                              alpha=alpha)

                    ax.set_xlim(0, .5)
                    ax0.set_xlim(0, 1)
                    for a in (ax, ax0):
                        a.text(-0.005, 1.03, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)")[i],
                               transform=a.transAxes, va='bottom', ha='right', fontweight='bold')
                        if i % 3:
                            a.set_ylabel("")
                            if i % 3 == 2:
                                a.text(
                                    1.05,  # X-coordinate (slightly outside the right boundary)
                                    0.5,  # Y-coordinate (centered vertically)
                                    f"${L_PHI_HOP}={phi_hop}$",  # The text to display
                                    rotation=-90,  # Rotate by 90 degrees
                                    va='center',  # Align vertically
                                    ha='left',  # Align horizontally
                                    transform=a.transAxes  # Use axes-relative coordinates
                                )
                        if i < 6:
                            a.set_xlabel("")
                            if i < 3:
                                a.text(
                                    .5,  # X-coordinate (slightly outside the right boundary)
                                    1.05,  # Y-coordinate (centered vertically)
                                    f"${L_PHI_H}={phi_H}$",  # The text to display
                                    rotation=0,  # Rotate by 90 degrees
                                    ha='center',
                                    va='bottom',
                                    transform=a.transAxes  # Use axes-relative coordinates
                                )
                    i += 1
            # savefig(f"sshh_butterfly_{label}")

        savefig("sshh_butterfly", f0)

        N = 50

        for phi_H, label, zoom in PHI_H:
            fig, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
            params = ParamsRange("phi_hop", (0, 2, .01), N=50, phi_H=phi_H, alpha=ALPHA)
            states = (params[80], 25), (params[120], 25), (params[80], 12), (params[120], 12)
            plotting.plot_eigvals(params=params, ax=ax, text="", states=states,
                                  xlabel=r"$\varphi_\mathrm{hop} / \frac{\pi}{4}$",
                                  xaxis=plotting.xaxis_getter("phi_hop"), do_compute_bands=DO_COMPUTE_BANDS)
            if zoom:
                rect = patches.Rectangle(
                    (0.02, .5), .8, ax.get_ylim()[1] - .55,
                    facecolor='white', alpha=.6, edgecolor='none', zorder=10
                )
                ax.add_patch(rect)
                ax.plot([.5*(zoom[0]+zoom[1])], [.5*(zoom[2]+zoom[3])], 's', markerfacecolor='none',
                        markersize=10, markeredgewidth=BOX_LINEWIDTH, markeredgecolor=BOX_EDGECOLOR)
                ax.add_patch(rect)
                inset_ax = ax.inset_axes((.12, .9, .4, .7), zorder=20, transform=ax.transData)
                inset_ax.yaxis.tick_right()
                inset_ax.set_xticks(zoom[:2])
                inset_ax.set_yticks(zoom[2:])
                for ticklabel in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                    ticklabel.set_fontsize("x-small")
                inset_params = ParamsRange("phi_hop", (zoom[0], zoom[1], .001),
                                           N=50, phi_H=phi_H, alpha=ALPHA)
                plotting.plot_eigvals(params=inset_params, ax=inset_ax, text="",
                                      xaxis=plotting.xaxis_getter("phi_hop"), xlabel=r"",
                                      ylabel="", do_compute_bands=DO_COMPUTE_BANDS)
                inset_ax.set_ylim(*zoom[2:])
            savefig(f"sshh_spectrum_{label}_phi_hop")
            _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
            for i, state in enumerate(states):
                ax = flatten_axes(axs)[i]
                x_pos = (.96, "right") if i % 2 else (.04, "left")
                plotting.plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]),
                                    color_index=i, show_xlabel=i == 3, show_ylabel=True, show_legend=False,
                                    ytick=(.3 if label=="insulator" else .2))
                ax.text(x_pos[0], .9, f"${L_PHI_HOP}={state[0].phi_hop:.1f}$",
                        ha=x_pos[1], va="top", fontsize="x-small", transform=ax.transAxes)
            savefig(f"sshh_states_{label}_phi_hop")

        for phi_hop, label, zoom in PHI_HOP:
            fig, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
            params = ParamsRange("phi_H", (0, 2, .01), N=50, phi_hop=phi_hop, alpha=ALPHA)
            i1 = 40 if phi_hop > 1 else 60
            states = (params[i1], 25), (params[120], 25), (params[i1], 2), (params[120], 2)
            plotting.plot_eigvals(params=params, ax=ax, text="", states=states,
                                  xlabel=r"$\varphi_\mathrm{H} / \frac{\pi}{4}$",
                                  xaxis=plotting.xaxis_getter("harper.phi_H"), do_compute_bands=DO_COMPUTE_BANDS)
            if zoom:
                rect = patches.Rectangle(
                    (0.02, .5), .8, ax.get_ylim()[1] - .55,
                    color='white', alpha=.6, edgecolor='none', zorder=10
                )
                ax.add_patch(rect)
                ax.plot([.5*(zoom[0]+zoom[1])], [.5*(zoom[2]+zoom[3])], 's', markerfacecolor='none',
                        markersize=10, markeredgewidth=BOX_LINEWIDTH, markeredgecolor=BOX_EDGECOLOR)
                ax.add_patch(rect)
                inset_ax = ax.inset_axes((.12, .9, .4, .7), zorder=20, transform=ax.transData)
                inset_ax.yaxis.tick_right()
                inset_ax.set_xticks(zoom[:2])
                inset_ax.set_yticks(zoom[2:])
                for ticklabel in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                    ticklabel.set_fontsize("x-small")
                inset_params = ParamsRange("phi_H", (zoom[0], zoom[1], .001),
                                           N=50, phi_hop=phi_hop, alpha=ALPHA)
                plotting.plot_eigvals(params=inset_params, ax=inset_ax, text="",
                                      xaxis=plotting.xaxis_getter("harper.phi_H"), xlabel=r"",
                                      ylabel="", do_compute_bands=DO_COMPUTE_BANDS)
                inset_ax.set_ylim(*zoom[2:])
            savefig(f"sshh_spectrum_{label}_phi_H")
            _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
            for i, state in enumerate(states):
                ax = flatten_axes(axs)[i]
                plotting.plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]),
                                    color_index=i, show_xlabel=i == 3, show_ylabel=True, show_legend=False,
                                    ytick=.3)
                ax.text(.96, .9, f"${L_PHI_H}={state[0].harper_not_none.phi_H:.1f}$",
                        ha="right", va="top", fontsize="x-small", transform=ax.transAxes)
            savefig(f"sshh_states_{label}_phi_H")


        for phi_hop, label, zoom in PHI_HOP:
            fig, axs = plt.subplots(2, 2, figsize=FS.BIG)
            for i, alpha in enumerate((F.GOLDEN_NUMBER, F.GOLDEN_NUMBER_REDUCED, F.SEC_MOST_IRRAT, F.PI)):
                params = ParamsRange("phi_H", (0, 2, .01), N=50, phi_hop=phi_hop,
                                     alpha=alpha)
                plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=get_label(alpha),
                                      xlabel=r"$\varphi_\mathrm{H} / \frac{\pi}{4}$",
                                      xaxis=plotting.xaxis_getter("harper.phi_H"), do_compute_bands=DO_COMPUTE_BANDS)
            savefig(f"sshh_spectrum_{label}_alpha")
        for phi_H, label, zoom in PHI_H:
            fig, axs = plt.subplots(2, 2, figsize=FS.BIG)
            for i, alpha in enumerate((F.GOLDEN_NUMBER, F.GOLDEN_NUMBER_REDUCED, F.SEC_MOST_IRRAT, F.PI)):
                params = ParamsRange("phi_hop", (0, 2, .01), N=50, phi_H=phi_H,
                                     alpha=alpha)
                plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=get_label(alpha),
                                      xlabel=r"$\varphi_\mathrm{hop} / \frac{\pi}{4}$",
                                      xaxis=plotting.xaxis_getter("phi_hop"), do_compute_bands=DO_COMPUTE_BANDS)
            savefig(f"sshh_spectrum_{label}_alpha")

        _, axs = plt.subplots(2, 2, figsize=FS.HUGE, sharex=True, sharey=True,
                              gridspec_kw={'hspace': 0.3})
        for i, (phi_hop, phi_H) in enumerate([(.75, .25), (1.25, .25), (.75, 1.25), (.75, .75)]):
            ax = flatten_axes(axs)[i]
            params = ParamsRange("beta", (0, 2*np.pi, np.pi/100),
                                 N=50, phi_H=phi_H, phi_hop=phi_hop, alpha=ALPHA)
            plotting.plot_eigvals(params=params, ax=ax, text="",
                                  xaxis=plotting.xaxis_getter("harper.beta", np.pi),
                                  xlabel=r"$\beta / \pi$", do_compute_bands=DO_COMPUTE_BANDS,
                                  )
            zoom = (.95, 1, .54, .58) if phi_H == .25 and phi_hop == .75 else None
            if zoom:
                rect = patches.Rectangle(
                    (0.02, -1.7), .8, 1.25,
                    facecolor='white', alpha=.6, edgecolor='none', zorder=10
                )
                ax.add_patch(rect)
                ax.plot([.5*(zoom[0]+zoom[1])], [.5*(zoom[2]+zoom[3])], 's', markerfacecolor='none',
                        markersize=10, markeredgewidth=BOX_LINEWIDTH, markeredgecolor=BOX_EDGECOLOR)
                ax.add_patch(rect)
                inset_ax = ax.inset_axes((.12, -1.6, .4, .7), zorder=20, transform=ax.transData)
                inset_ax.xaxis.tick_top()
                inset_ax.yaxis.tick_right()
                inset_ax.set_xticks(zoom[:2])
                inset_ax.set_yticks(zoom[2:])
                for ticklabel in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                    ticklabel.set_fontsize("x-small")
                precision = .001
                inset_params = ParamsRange("beta", (np.pi*zoom[0], np.pi*zoom[1], np.pi*precision),
                                     N=50, phi_H=phi_H, alpha=ALPHA, phi_hop=phi_hop)
                plotting.plot_eigvals(params=inset_params, ax=inset_ax, text="",
                                      xaxis=plotting.xaxis_getter("harper.beta", np.pi), xlabel=r"",
                                      ylabel="", do_compute_bands=DO_COMPUTE_BANDS)
                inset_ax.set_ylim(*zoom[2:])
                inset_ax.axvline(((-51*ALPHA) % 1), color="gray", linewidth=.5)

            ax.set_title(f"${L_PHI_H}={phi_H}$, ${L_PHI_HOP}={phi_hop}$")
            ax.text(-0.05, 1.05, ("(a)", "(b)", "(c)", "(d)")[i], transform=ax.transAxes, va='bottom', ha='right',
                    fontweight='bold')
            ax.set_ylim(-1.75, 1.75)
            if i not in (0, 2):
                ax.set_ylabel("")
            if i not in (2, 3):
                ax.set_xlabel("")
        savefig("sshh_spectrum_beta")

## Localization Harper

    if do_plot["localization_harper"]:

        _, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=Figsize.HUGE)

        phi_H_grid = np.arange(0, 2.0001, 0.01)
        alphas = [Harper.IrrationalFrequency.GOLDEN_NUMBER]
        alphas_special = []


        def params_getter_harper(_N: int):
            return lambda alpha: [
                ChainModel.Params(N=_N, harper=Harper(phi_H=phi_H, alpha=alpha)) for phi_H in phi_H_grid
            ]

        for i, N in enumerate([50, 200, 1000, 5000]):
            ax = flatten_axes(axs)[i]
            show_legend = i == 0
            plotting.plot_localization(params_getter_harper(N), ax=ax, alphas=alphas,
                                       xlabel=r"$\varphi_H / \frac{\pi}{4}$" * (i > 1),
                                       ylabel="$L$" * ((i + 1) % 2), show_legend=show_legend,
                                       alphas_special=alphas_special)

        savefig("localization_harper")


        _, ax = plt.subplots(figsize=Figsize.NORMAL)
        N= 1000
        phi_H_grid = np.arange(0, 2.0001, 0.01)

        alphas = F.GOLDEN_NUMBER, F.GOLDEN_NUMBER_REDUCED, F.SEC_MOST_IRRAT, F.PI, F.SQRT_2_NORMALIZED, Harper.Frequency(.65)
        alphas_special = []

        plotting.plot_localization(params_getter_harper(N), ax=ax, alphas=alphas,
                                   xlabel=r"$\varphi_H / \frac{\pi}{4}$",
                                   ylabel="$L$" * ((i + 1) % 2), show_legend="methods_alphas",
                                   alphas_special=alphas_special, show_title=False,
                                   plot_num_every=10)

        savefig("localization_harper_alphas")

## Localization SSH

    if do_plot["localization_ssh"]:
        N = 5000
        params = ParamsRange("phi_hop", (0.1, 1, .01), N=N)
        LocalizationMeasure.measure_systems_all_methods(params)
        for method in LocalizationMeasure.Method:
            localization_measure = LocalizationMeasure(method=method)
            minmaxmean: tuple[list, list, list] = [], [], []
            for p in params:
                localization = localization_measure.measure_all(p)
                localization[N//2-1:N//2+1] = np.nan
                minmaxmean[0].append(np.nanmin(localization))
                minmaxmean[1].append(np.nanmax(localization))
                minmaxmean[2].append(np.nanmean(localization))
            result_logger.info("Localization pure SSH with measure %s without edge states "
                               "min %.5f +/- %.5f max %.5f +/- %.5f mean %.5f +/- %.5f",
                               method, np.mean(minmaxmean[0]), np.std(minmaxmean[0]), np.mean(minmaxmean[1]),
                               np.std(minmaxmean[1]), np.mean(minmaxmean[2]), np.std(minmaxmean[2]))


## Localization SSHH

    if do_plot["localization_sshh"]:

        localization_label = r"$\bar{L}_\mathrm{bin}$"

        for phi_hop in [.8]:  # (.2, .4, .6, .8):
            for alpha in [a for a in F if a.name.startswith("GOLDEN_NUMBER_1_") or a==F.GOLDEN_NUMBER_REDUCED]:

                _, axs = plt.subplots(2, 1, figsize=Figsize.BIG, sharex=True)
                params = ParamsRange("phi_H", (0, 2, .01),
                                     N=5000, phi_hop=phi_hop, alpha=alpha)
                localization_measure = LocalizationMeasure(method=LocalizationMeasure.Method.BIN)
                localization = localization_measure.measure_systems(params)
                steps = .465, .54, .64, .74, .95
                for step in steps:
                    for ax in flatten_axes(axs):
                        ax.axvline(step, color="black", linewidth=.5)
                flatten_axes(axs)[0].plot(params.param_values, localization, label="BIN", color="black")
                flatten_axes(axs)[0].set_ylabel(localization_label)
                flatten_axes(axs)[0].set_yticks([0, .5, 1])
                plotting.plot_eigvals_localization_per_band(params,
                                                            LocalizationAnalyzer(band_getter=BandGetter(),
                                                                                 method=LocalizationMeasure.Method.BIN),
                                                            ax=flatten_axes(axs)[1],
                                                            xlabel=r"$\varphi_H / \frac{\pi}{4}$",
                                                            clabel=localization_label)
                ax = flatten_axes(axs)[0]
                inset_ax = ax.inset_axes((1.4, .2, .5, .6), zorder=20, transform=ax.transData)
                xlim = .6, .7
                ylim = .54, .77
                zoom  = xlim[0], xlim[1], ylim[0], ylim[1]
                box = patches.Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                        facecolor='none', edgecolor="black", zorder=20, linewidth=BOX_LINEWIDTH)
                flatten_axes(axs)[1].add_patch(box)
                inset_params = ParamsRange("phi_H", (*xlim, .002),
                                     N=5000, phi_hop=phi_hop, alpha=alpha)
                plotting.plot_eigvals_localization_per_band(inset_params,
                                                            LocalizationAnalyzer(band_getter=BandGetter(),
                                                                                 method=LocalizationMeasure.Method.BIN),
                                                            ax=inset_ax,
                                                            xlabel="", ylabel="")
                inset_ax.set_xlim(*xlim)
                inset_ax.set_ylim(*ylim)
                savefig(f"localization_sshh_spectrum_{alpha.name}_{phi_hop}")


## Edgestates Harper

    if do_plot["edgestates_harper"]:

        def plot_edge_states(fig_label: str, _N:int, _box: tuple, _phi_H: float, _phi_hop: float,
                             _beta: float | None = None):

            f_specs, axs_specs = plt.subplots(1, 2, figsize=FS.TWO_THIRDS_NORMAL, sharey=True,
                                  gridspec_kw={'wspace': 0.01})
            f_states, axs_states = plt.subplots(4, 2, sharey=True, figsize=FS.ONE_THIRD_NORMAL,
                                                gridspec_kw={'wspace': 0.01})

            plotting.plot_edge_states_alpha(_box, axs_specs, axs_states, _N, _phi_H, _phi_hop, _beta)

            savefig(f"edgestates_{fig_label}_states", f_states)
            savefig(f"edgestates_{fig_label}_spectrum", f_specs)


        for beta in (0, 1):
            _, axs = plt.subplots(2, 2, figsize=FS.BIG, sharex=True, sharey=True)
            for i, N_or_None in enumerate((50, 54, 56, 100)):


                params = ParamsRange("phi_H", (0, 2, .01), N=(N_or_None or 10),
                                     alpha=F.GOLDEN_NUMBER, beta=beta)
                plotting.plot_eigvals(params=params, ax=flatten_axes(axs)[i], text=f"$N={N_or_None}$" if N_or_None else "",
                                      ylim=(None, None),
                                      xlabel=r"$\varphi_H / \frac{\pi}{4}$"*(i>1),
                                      ylabel="$E$"*((i+1)%2),
                                      do_compute_bands=DO_COMPUTE_BANDS,
                                      plot_spectrum=bool(N_or_None),
                                      edge_state_threshold=0,
                                      edge_state_max_count=1)
            savefig(f"edgestates_harper_spectrum_N_{beta}")


            phi_Hs = {"metallic": .8, "insulator": 1.2}
            for label, phi_H in phi_Hs.items():

                params = ParamsRange("beta", (0, 2 * np.pi, np.pi / 100),
                                     N=50, phi_H=phi_H, alpha=F.GOLDEN_NUMBER)
                p = params[140]
                assert p.harper.beta == 1.4 * np.pi, p.harper.beta / np.pi
                _, ax = plt.subplots(figsize=FS.HALF_NORMAL)
                plotting.plot_eigvals(params=params, ax=ax, text="",
                                      xaxis=plotting.xaxis_getter("harper.beta", np.pi), xlabel=r"$\beta / \pi$",
                                      do_compute_bands=DO_COMPUTE_BANDS,
                                      edge_state_threshold=.5 if label == "metallic" else .9)
                savefig(f"edgestates_harper_beta_{label}")


        N = 50
        alpha = .8
        d_alpha = .001
        ms = 1
        est = .95
        zoom = .6, .7, 0, 1

        for beta in (0, 1):

            for factor, label in ((None, "commensurable"), (F.GOLDEN_NUMBER, "incommensurable")):

                if factor is None:
                    params = ParamsRange("alpha", (d_alpha, 1-d_alpha, d_alpha), phi_H=1, N=N, beta=beta)
                    inset_params = ParamsRange("alpha", (zoom[0], zoom[1], d_alpha/10), phi_H=1, N=N, beta=beta)
                elif isinstance(factor, Harper.IrrationalFrequency):
                    params = ParamsRange("alpha", (d_alpha, d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=1, N=N, beta=beta)
                    inset_params = ParamsRange("alpha", (d_alpha * (int(zoom[0]/d_alpha/factor) + 1), d_alpha * (int(zoom[1]/d_alpha/factor) - 1), d_alpha/10, factor), phi_H=1, N=N, beta=beta)
                else:
                    raise ValueError(factor)
                _, ax = plt.subplots(figsize=FS.HALF_NORMAL)
                plotting.plot_eigvals(params=params, ax=ax, text="", do_compute_bands=False,
                                      xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                                      line_style=" ", marker="o", markersize=ms, markerfacecolor="black", markeredgewidth=0,
                                      alpha=alpha, ylim=(-2, 2), edge_state_threshold=est)
                ax.set_xlim(0, 1)

                rect = patches.Rectangle(
                    (0.64, -1.9), .34, 1.2,
                    facecolor='white', alpha=.6, edgecolor='none', zorder=10
                )
                ax.add_patch(rect)
                box = patches.Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                        facecolor='none', edgecolor=BOX_EDGECOLOR, zorder=20, linewidth=BOX_LINEWIDTH)
                ax.add_patch(box)
                inset_ax = ax.inset_axes((.66, -1.8, .3, 1), zorder=20, transform=ax.transData)
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                for ticklabel in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
                    ticklabel.set_fontsize("x-small")
                plotting.plot_eigvals(params=inset_params, ax=inset_ax, text="", do_compute_bands=False,
                                      xaxis=plotting.xaxis_getter("harper.alpha"), xlabel="", ylabel="",
                                      line_style=" ", marker="o", markersize=ms, markerfacecolor="black", markeredgewidth=0,
                                      alpha=alpha, ylim=(zoom[2], zoom[3]), edge_state_threshold=est)
                inset_ax.set_xlim(zoom[:2])

                savefig(f"edgestates_harper_butterfly_{label}_{beta}")

## Edgestates SSHH

    if do_plot["edgestates_sshh"]:

        _, ax = plt.subplots(figsize=FS.NORMAL)
        N = 200
        phi_H = .2
        phi_hop = 1.3
        beta = 0
        alpha = .8  # opacity
        ms = .4
        d_alpha = 0.001

        for factor, label in ((None, "commensurable"), (F.GOLDEN_NUMBER, "incommensurable")):

            if factor is None:
                params = ParamsRange("alpha", (d_alpha, .5 - d_alpha, d_alpha),
                                      phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
            elif isinstance(factor, Harper.IrrationalFrequency):
                params = ParamsRange("alpha", (
                    d_alpha * (int(1 / d_alpha / factor) // 2 + 1), d_alpha * (int(1 / d_alpha / factor) - 1),
                    d_alpha, factor), phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
            else:
                raise ValueError(factor)

            plotting.plot_eigvals(params=params, ax=ax, do_compute_bands=False, text="",
                                  xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                                  line_style=" ", marker="o", markersize=ms, markerfacecolor="black",
                                  markeredgewidth=0,
                                  alpha=alpha,
                                  edge_state_threshold=.5)

        ax.set_xlim(0, 1)
        savefig("edgestates_sshh_butterfly")



        N = 50
        PHI_HOPS = .75, 1.25
        PHI_HS = .4, 1.1

        beta_0 = 1.4 * np.pi

        states = []

        def add_states(_N, _phi_hop, _phi_H, _alpha, n):
            _p = ChainModel.Params(N=_N, phi_hop=_phi_hop, harper=Harper(phi_H=_phi_H, alpha=_alpha, beta=beta_0))
            _loc_vals = LocalizationMeasure(method=LocalizationMeasure.Method.EDGE).measure_all(_p)
            _new_states = [(_p, int(idx)+1) for idx in np.argsort(_loc_vals)[:n]]
            states.extend(_new_states)
            return _new_states

        _, axs = plt.subplots(2, 2, figsize=FS.BIG, sharex=True, sharey=True)
        plt.subplots_adjust(right=0.8)
        for i, phi_hop in enumerate(PHI_HOPS):
            for j, phi_H in enumerate(PHI_HS):
                idx = i + 2*j
                ax = flatten_axes(axs)[idx]
                edge_state_threshold = .5 if j==0 else .9
                new_states = add_states(N, phi_hop, phi_H, F.GOLDEN_NUMBER, 2 if i==1 and j==0 else 1)
                params = ParamsRange("beta", (0, 2 * np.pi, np.pi / 100),
                                     N=50, phi_H=phi_H, alpha=F.GOLDEN_NUMBER, phi_hop=phi_hop)
                plotting.plot_eigvals(params=params, ax=ax, text="",
                                      xaxis=plotting.xaxis_getter("harper.beta", np.pi),
                                      xlabel=r"$\beta / \pi$" * j,
                                      ylabel="E" * (i==0),
                                      do_compute_bands=DO_COMPUTE_BANDS,
                                      states=new_states*0,
                                      edge_state_threshold=edge_state_threshold)

                ax.text(-0.005, 1.03, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)")[idx],
                       transform=ax.transAxes, va='bottom', ha='right', fontweight='bold')
                if i == 1:
                    ax.text(
                        1.05,  # X-coordinate (slightly outside the right boundary)
                        0.5,  # Y-coordinate (centered vertically)
                        f"${L_PHI_H}={phi_H}$",  # The text to display
                        rotation=-90,  # Rotate by 90 degrees
                        va='center',  # Align vertically
                        ha='left',  # Align horizontally
                        transform=ax.transAxes  # Use axes-relative coordinates
                    )
                if j == 0:
                    ax.text(
                        .5,  # X-coordinate (slightly outside the right boundary)
                        1.05,  # Y-coordinate (centered vertically)
                        f"${L_PHI_HOP}={phi_hop}$",  # The text to display
                        rotation=0,  # Rotate by 90 degrees
                        ha='center',
                        va='bottom',
                        transform=ax.transAxes  # Use axes-relative coordinates
                    )
        savefig("edgestates_sshh_beta")


        _, axs = plt.subplots(len(states), 1, figsize=FS.NORMAL)
        for i, ax in enumerate(flatten_axes(axs)):
            plotting.plot_state(states[i], ax=ax, plot_probability=True, show_legend=False,
                                psi_index=str(states[i][1]),
                                show_xlabel=i==len(states)-1, show_ylabel=True,
                                color_index=i)
        savefig("edgestates_sshh_beta_states")


        d_alpha = 0.01
        alpha = .8
        ms = .4


        for beta in (0, 1):

            f0, axs0 = plt.subplots(3, 3, figsize=FS.HUGE_QUADRATIC, sharex=True, sharey=True,
                                  gridspec_kw={})
            plt.subplots_adjust(right=0.8)

            for factor, label in ((None, "commensurable"), (F.GOLDEN_NUMBER, "incommensurable")):


                _, axs = plt.subplots(3, 3, figsize=FS.HUGE_QUADRATIC, sharex=True, sharey=True,
                                      gridspec_kw={})
                plt.subplots_adjust(right=0.8)
                i = 0
                for phi_hop in (.7, 1.3, 1.6):
                    for phi_H in (.2, .5, 1):
                        ax = flatten_axes(axs)[i]
                        ax0 = flatten_axes(axs0)[i]

                        if factor is None:
                            params = ParamsRange("alpha", (d_alpha, 1 - d_alpha, d_alpha), phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
                            params0 = ParamsRange("alpha", (d_alpha, .5 - d_alpha, d_alpha), phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
                        elif isinstance(factor, Harper.IrrationalFrequency):
                            params = ParamsRange("alpha", (d_alpha, d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
                            params0 = ParamsRange("alpha", (d_alpha * (int(1/d_alpha/factor)//2 + 1), d_alpha * (int(1/d_alpha/factor) - 1), d_alpha, factor), phi_H=phi_H, phi_hop=phi_hop, N=N, beta=beta)
                        else:
                            raise ValueError(factor)

                        for a, p in ((ax, params), (ax0, params0)):
                            plotting.plot_eigvals(params=p, ax=a, do_compute_bands=False, text="",
                                                  xaxis=plotting.xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                                                  line_style=" ", marker="o", markersize=ms, markerfacecolor="black",
                                                  markeredgewidth=0,
                                                  alpha=alpha,
                                                  edge_state_threshold=.9)

                        ax.set_xlim(0, .5)
                        ax0.set_xlim(0, 1)
                        for a in (ax, ax0):
                            a.text(-0.005, 1.03, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)")[i],
                                   transform=a.transAxes, va='bottom', ha='right', fontweight='bold')
                            if i % 3:
                                a.set_ylabel("")
                                if i % 3 == 2:
                                    a.text(
                                        1.05,  # X-coordinate (slightly outside the right boundary)
                                        0.5,  # Y-coordinate (centered vertically)
                                        f"${L_PHI_HOP}={phi_hop}$",  # The text to display
                                        rotation=-90,  # Rotate by 90 degrees
                                        va='center',  # Align vertically
                                        ha='left',  # Align horizontally
                                        transform=a.transAxes  # Use axes-relative coordinates
                                    )
                            if i < 6:
                                a.set_xlabel("")
                                if i < 3:
                                    a.text(
                                        .5,  # X-coordinate (slightly outside the right boundary)
                                        1.05,  # Y-coordinate (centered vertically)
                                        f"${L_PHI_H}={phi_H}$",  # The text to display
                                        rotation=0,  # Rotate by 90 degrees
                                        ha='center',
                                        va='bottom',
                                        transform=a.transAxes  # Use axes-relative coordinates
                                    )
                        i += 1
                # savefig(f"sshh_butterfly_{label}")

            savefig(f"edgestates_sshh_butterfly_{beta}", f0)

        N = 200
        for beta in (((-2*np.pi*F.GOLDEN_NUMBER_REDUCED + np.pi/2) % 2*np.pi), np.pi/2, np.pi, 0):
            for phi_H in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                _, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
                params = ParamsRange( "phi_hop", (0, 2, .01),
                                     N=200, alpha=F.GOLDEN_NUMBER_REDUCED, beta=beta, phi_H=phi_H)
                x, y, z = 100, 2, 10
                states = []  # (params[x-z], 2500), (params[x-y], 2500), (params[x+y], 2500), (params[x+z], 2500)
                plotting.plot_eigvals(params, xaxis=plotting.xaxis_getter("phi_hop"), ax=ax, do_compute_bands=False,
                                      ylim=(-.2, .2), xlabel="phi_hop",
                                      edge_state_threshold=.9, states=states)
                savefig(f"edgestates_sshh_eye3_{N}_{beta}_{phi_H}")
                _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
                for i, state in enumerate(states):
                    plotting.plot_state(state, ax=flatten_axes(axs)[i], plot_probability=True, psi_index=str(state[1]),
                                        color_index=i, show_xlabel=i == 3, show_ylabel=True, show_legend=False)

                savefig(f"edgestates_sshh_eye3_states_{N}_{beta}_{phi_H}")


        _, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
        params = ParamsRange( "phi_H", (0.8, 1.1, .001),
                             N=5000, alpha=F.GOLDEN_NUMBER_REDUCED, beta=0, phi_hop=1.25)
        x, y, z = 69, 2, 12
        states = (params[x-z], 2500), (params[x-y], 2500), (params[x+y], 2500), (params[x+z], 2500)
        plotting.plot_eigvals(params, xaxis=plotting.xaxis_getter("harper.phi_H"), ax=ax, do_compute_bands=False,
                              ylim=(-.02, .02),
                              edge_state_threshold=None, states=states)
        savefig("edgestates_sshh_eye")
        _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
        for i, state in enumerate(states):
            plotting.plot_state(state, ax=flatten_axes(axs)[i], plot_probability=True, psi_index=str(state[1]),
                                color_index=i, show_xlabel=i == 3, show_ylabel=True, show_legend=False)

        savefig("edgestates_sshh_eye_states")



        _, ax = plt.subplots(figsize=FS.TWO_THIRDS_NORMAL)
        params = ParamsRange( "phi_hop", (0, 2, .01),
                             N=5000, alpha=F.GOLDEN_NUMBER_REDUCED, beta=0, phi_H=.9)
        x, y, z = 100, 2, 10
        states = (params[x-z], 2500), (params[x-y], 2500), (params[x+y], 2500), (params[x+z], 2500)
        plotting.plot_eigvals(params, xaxis=plotting.xaxis_getter("phi_hop"), ax=ax, do_compute_bands=False,
                              ylim=(-.02, .02), xlabel="phi_hop",
                              edge_state_threshold=None, states=states)
        savefig("edgestates_sshh_eye2")
        _, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=FS.ONE_THIRD_NORMAL)
        for i, state in enumerate(states):
            plotting.plot_state(state, ax=flatten_axes(axs)[i], plot_probability=True, psi_index=str(state[1]),
                                color_index=i, show_xlabel=i == 3, show_ylabel=True, show_legend=False)

        savefig("edgestates_sshh_eye2_states")

