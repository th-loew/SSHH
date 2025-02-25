from typing import Iterable, Callable, Any, Literal
from collections.abc import Sized, Sequence

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bands import BandGetter, iterate_bands
from chain_model import ChainModel
from harper import Harper, beta_symmetric_harper_hamiltonian
from localization_analyzer import LocalizationAnalyzer
from localization_measure import LocalizationMeasure
from params_range import ParamsRange, ParamsArray2D
from plotting_utils import HandlerLineCollectionShowAll, get_color, flatten_axes, get_label, COLORS
from utils import FloatArray

CMAP = "YlOrRd"

StateDescriptor = int
State = tuple[ChainModel.Params, StateDescriptor]
States = Iterable[State] | Iterable[Iterable[State]]

_colors = [np.asarray(to_rgb(h)) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def get_state_index(state_descriptor: StateDescriptor) -> int:
    return state_descriptor - 1


def xaxis_getter(fields: str | Iterable[str], normalization: float | None = None) \
        -> Callable[[ChainModel.Params], float]:
    field_list = fields.split(".") if isinstance(fields, str) else list(fields)
    def get_xaxis_value(params: ChainModel.Params) -> float:
        value = params
        for field in field_list:
            value = getattr(value, field)
        assert isinstance(value, (int, float)), "Value must be int or float but is " + str(type(value))
        return float(value) / normalization if normalization else float(value)
    return get_xaxis_value


def _flatten_states(states: States) -> tuple[int, list[State]]:
    if not states:
        return 1, list()
    states_flat: list[State] = []
    sublens: list[int] = []
    for s in states:
        if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], ChainModel.Params) \
                and isinstance(s[1], StateDescriptor):
            states_flat.append(s)
            sublens.append(1)
        else:
            sublen1 = 0
            for s1 in s:
                if isinstance(s1, tuple) and len(s1) == 2 and isinstance(s1[0], ChainModel.Params) \
                        and isinstance(s1[1], StateDescriptor):
                    states_flat.append((s1[0], s1[1]))
                    sublen1 += 1
                else:
                    raise ValueError("Invalid state")
            sublens.append(sublen1)
    sublen = sublens[0]
    assert all(sl==sublen for sl in sublens), "All sublists must have the same length"
    return sublen, states_flat


def get_eigvals(params: Iterable[ChainModel.Params]):
    params = tuple(params)
    eigvals = np.zeros((len(params), 2*params[0].M))
    for i, param in enumerate(params):
        ssh = ChainModel(param)
        eigvals[i, :] = ssh.eigh(eigvals_only=True)
    return eigvals

def plot_eigvals(params: ParamsRange,
                 xaxis: Callable[[ChainModel.Params], float] = xaxis_getter("harper.phi_H"),
                 xlabel: str = r"$\varphi_H / \frac{\pi}{4}$",
                 states: States = tuple(),
                 ylim: tuple[float | None, float | None] = (None, None),
                 text: str | None = None,
                 ax: plt.Axes | None = None,
                 ylabel: str = "$E$",
                 do_compute_bands: bool = False,
                 line_style: str = "-k",
                 plot_spectrum: bool = True,
                 state_marks: Sequence[str] | None = None,
                 state_color_idx: Sequence[int] | None = None,
                 edge_state_threshold: float | Sequence[float] | None = None,
                 edge_state_max_count: int = -1,
                 **plot_params
                 ):
    plot_params.setdefault("linewidth", 0.5)

    if not params:
        raise ValueError("No parameters given")
    assert len(set(xaxis(p) for p in params)) == len(params), "xaxis values must be unique"
    _, states = _flatten_states(states)
    M = params[0].M
    if not all(param.M == M for param in params):
        raise ValueError("All parameters must have the same N")
    x = np.asarray([xaxis(param) for param in params])

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10)

    if params.param == "beta" and all(ChainModel(p).name == "Harper" for p in params):
        betas = beta_symmetric_harper_hamiltonian(params[0].harper.alpha, params[0].N)
        for beta in betas:
            ax.axvline(beta/np.pi, color="gray", linewidth=.5)

    band_getter = BandGetter(do_not_compute=not do_compute_bands)
    try:
        bands = band_getter.get_nonan_bands_params_range(params)
    except band_getter.LoadExceptionDoNotCompute:
        pass
    else:
        for i in range(bands.shape[0] // 2):
            ax.fill_between(x, bands[2*i], bands[2*i+1], color='lightblue', linewidth=0)

    if plot_spectrum:
        eigvals = get_eigvals(params)
        ax.plot(x, eigvals, line_style, **plot_params)

    for i, (p, sd) in enumerate(states):
        model = ChainModel(p)
        x_state, y_state = xaxis(p), float(model.eigh(eigvals_only=True)[get_state_index(sd)])
        if state_marks:
            mark = state_marks[i]
            mew = 1
        else:
            mark = "o"
            mew = 2
        if state_color_idx:
            color = _colors[state_color_idx[i]]
        else:
            color = _colors[i]
        ax.plot(x_state, y_state, mark, markeredgecolor=color, markerfacecolor=(0, 0, 0, 0), markeredgewidth=mew, markersize=2+4*plot_params["linewidth"])
        #ax.plot(x_state, y_state, '.', color='red', markersize=16)
        #ax.text(x_state, y_state, f"{i+1}", verticalalignment='center', horizontalalignment='center', color='white',
        #        fontsize=8)

    if edge_state_threshold is not None:

        est = edge_state_threshold

        loc_vals = LocalizationMeasure(method=LocalizationMeasure.EDGE_METHOD).measure_systems(params, modus="copy").T
        for m_sign in (-1, 1):
            c = COLORS[2] if m_sign > 0 else COLORS[3]
            e = eigvals.copy().T
            for j in range(e.shape[1]):
                m = m_sign * (est if isinstance(est, (float, int)) else est[j])
                M = np.sort(loc_vals[:, j])[::-m_sign][edge_state_max_count-1]
                m = max(m, M) if m_sign > 0 else min(m, M)
                idx = (loc_vals[:, j] > m) if m_sign < 0 else (loc_vals[:, j] < m)
                e[idx, j] = np.nan
            ax.plot(x, e.T, ".", label="edge state",
                    markeredgecolor="none", markerfacecolor=c, markersize=3)


    if text:
        ax.text(0.05, 0.85, text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7), verticalalignment='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*ylim)


def plot_eigvals_localization_per_band(params: ParamsRange,
                                       localization_analyzer: LocalizationAnalyzer,
                                       xlabel: str = r"$\varphi_H / \frac{\pi}{4}$",
                                       ylim: tuple[float | None, float | None] = (None, None),
                                       text: str | None = None,
                                       ax: plt.Axes | None = None,
                                       ylabel: str = "$E$",
                                       clabel: str = "",
                                       ):

    x = params.param_values
    cmap = LinearSegmentedColormap.from_list("localization", [COLORS[0], np.array([0, 0, 0]), COLORS[3]])
    norm = Normalize(0, 1)
    bands, loc_vals = localization_analyzer.analyze_bands(params)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10)

    linewidth = 1.6

    for j, xj in enumerate(x):
        for (i, _), (lower, upper) in iterate_bands(bands[:, j]):
            loc_val = loc_vals[i, j]
            color = cmap(norm(loc_val))
            ax.plot((xj, xj), (lower, upper), color=color, linewidth=linewidth)


    if text:
        ax.text(0.05, 0.85, text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*ylim)

    if clabel and isinstance(ax.figure, Figure):
        fig = ax.figure
        cbar_ax = fig.add_axes((0.92, 0.108, 0.03, 0.772))  # [left, bottom, width, height]
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label(clabel)

def plot_state(state: State, ax: plt.Axes, max_cells=None, show_xlabel=True, plot_probability=False,
               psi_index: str | None = None, color_index: int | None = None, show_ylabel=True,
               show_legend: bool = True, set_ticks=True, ytick: float | None = None) -> tuple[float, float]:

    params, state_descriptor = state
    has_AB = params.phi_hop != 1
    if psi_index is None:
        psi_index = str(state_descriptor)
    ssh = ChainModel(params)
    eigvals, eigvecs = ssh.eigh()

    idx = get_state_index(state_descriptor)
    eigval = eigvals[idx]

    values = np.abs(eigvecs)**2 if plot_probability else eigvecs

    values_A = values[:, idx].copy()
    values_A[1::2] = 0
    values_B = values[:, idx].copy()
    values_B[::2] = 0

    if has_AB and color_index is not None:
        values_A = values_A[::2]
        values_B = -values_B[1::2]

    if abs(eigval) < 1e-3 and not plot_probability:
        #maxval = max(abs(values_A[0]), abs(values_B[-1]))
        x = np.linspace(0, params.N-1, 100)
        #ax.fill_between(1 + x / 2,
        #                maxval * np.exp(-x / ssh.SSH_localization_length / 2),
        #                -maxval * np.exp(-x / ssh.SSH_localization_length / 2),
        #                color='lightgray')
        #ax.fill_between(.5 + (params.N-1-x) / 2, maxval * np.exp(-x / ssh.SSH_localization_length / 2),
        #                -maxval * np.exp(-x / ssh.SSH_localization_length / 2),
        #                color='lightgrey')

    color_kwargs: dict[str, Any] = {'color': _colors[color_index]} if color_index is not None \
        else {} if has_AB else {'color': 'black'}
    width = 0.5 if has_AB and color_index is None else 1
    x = (.75 + np.arange(params.N)/2) if (has_AB and color_index is None) \
        else (1.0 + np.arange(params.M)) if has_AB \
        else (1.0 + np.arange(params.N))
    #if abs(eigval) < 1e-3 and plot_probability and color_index:
    #    maxval = max([abs(values_A[0]), abs(values_B[0]), abs(values_A[-1]), abs(values_B[-1])])
    #    x_extended = np.concatenate(([2*x[0] - x[1]], x, [2*x[-1] - x[-2]]))
    #    for a in (x[0]-x_extended, x_extended-x[-1]):
    #        y = maxval * np.exp(a / ssh.SSH_localization_length * 2)
    #        # fill between without line
    #        ax.fill_between(x_extended, y, -y, color='lightgray', linewidth=0, step='mid')

    ax.bar(x, values_A, label="A", width=width, **color_kwargs)
    ax.bar(x, values_B, label="B", width=width, **color_kwargs)
    ylabel = r'\psi_{' + psi_index + '}'
    if plot_probability:
        ylabel = '|' + ylabel + '|^2'
    if show_ylabel:
        ax.set_ylabel('$' + ylabel + '$')
    if show_xlabel:
        ax.set_xlabel(r'unit cell $m$' if has_AB else r'lattice site $n$')
    if has_AB and show_legend:
        ax.legend(bbox_to_anchor=(1.15, 1))
    if max_cells is not None:
        ax.set_xlim(min(x) - width/2, min(max(x) + width/2, max_cells))
    else:
        ax.set_xlim(min(x) - width/2, max(x) + width/2)
    if has_AB and color_index and set_ticks:
        assert plot_probability
        set_state_AB_yticks(ax, max(values_A), max(-values_B), ytick=ytick)
    return max(values_A), max(-values_B)

def set_state_AB_yticks(ax: plt.Axes, max_A: float, max_B: float, ytick: float | None = None):
    if ytick is None:
        yticks = -.2, 0, .2
        for val in (.5, .4, .3):
            if max_A > val and max_B > val:
                yticks = -val, 0, val
    else:
        yticks = -ytick, 0, ytick
    ytick_labels = tuple(f"{y} A" if y > 0 else f"{-y} B" if y < 0 else "" for y in yticks)
    ax.set_yticks(yticks, labels=ytick_labels)


def plot_states(states: States = tuple(), max_cells: int | None = None, sharey: bool = False,
                plot_probability: bool = False, axs: Sequence[plt.Axes] | None = None):

    sublen, states = _flatten_states(states)

    if axs is None:
        fig, _axs = plt.subplots(len(states)//sublen, sublen, sharex=True, sharey=sharey)
        fig.set_size_inches(10, 1.5 * len(states))
        axs = flatten_axes(_axs)

    max_A, max_B = .0, .0

    for i, state in enumerate(states):
        ax: plt.Axes = axs[i]
        m_A, m_B = plot_state(state, ax, max_cells=max_cells, show_xlabel=(i == len(states) - 1),
                              plot_probability=plot_probability, psi_index=str(i+1), set_ticks=False)
        max_A = max(max_A, m_A)
        max_B = max(max_B, m_B)
    for ax in axs:
        set_state_AB_yticks(ax, max_A, max_B)


def plot_states_oneplot(states: States | ChainModel.Params,
                        yaxis: Callable[[ChainModel.Params], float] | str | None = None,
                        ylabel: str | None = None,
                        show_legend: bool = True,
                        ax: plt.Axes | None = None,
                        colored: bool | None = None,
                        ):

    show_legend = show_legend and bool(colored)

    if ax is None:
        _, ax = plt.subplots()

    color_A = _colors[0][0], _colors[0][1], _colors[0][2]
    color_B = _colors[1][0], _colors[1][1], _colors[1][2]
    color_anti = _colors[2][0], _colors[2][1], _colors[2][2]
    color_sym = _colors[4][0], _colors[4][1], _colors[4][2]

    def _get_color(A: float, B: float, colored: bool, max_prob: float = 1) -> tuple[float, float, float, float]:
        alpha = (A**2 + B**2) / max_prob
        if not colored:
            return 0, 0, 0, alpha
        color_2 = np.array(color_anti if A * B < 0 else color_sym)
        color_1 = np.array(color_A if abs(A) > abs(B) else color_B)
        p = 0 if abs(A) == abs(B) else abs(abs(A) - abs(B)) / max(abs(A), abs(B))
        color = tuple(float(x) for x in color_1 * p + color_2 * (1 - p))
        return color[0], color[1], color[2], alpha

    if isinstance(states, ChainModel.Params):
        colored = colored if colored is not None else states.phi_hop != 1
        N = states.M
        n_states = 2*N
        chain_model = ChainModel(states)
        _, eigvecs = chain_model.eigh()
        assert yaxis is None and ylabel is None
        ylabel = "quantum number $n$"
        yticks = list(range(0, 2*N, (2*N)//5))
        ax.set_yticks(yticks, labels=[str(y) for y in yticks])
    else:
        _, states = _flatten_states(states)
        colored = colored if colored is not None else any(p.phi_hop != 1 for p, _ in states)
        n_states = len(states)
        N = states[0][0].M
        eigvecs = np.zeros((2*N, len(states)))
        for i, (params, n) in enumerate(states):
            chain_model = ChainModel(params)
            eigvecs[:, i] = chain_model.eigh(eigvals_only=False)[1][:, n]
        assert yaxis is not None and ylabel is not None
        yticks = list(range(0, n_states, n_states//5)) if n_states > 10 else list(range(n_states))
        yaxis_fun = (lambda p: getattr(p, yaxis)) if isinstance(yaxis, str) else yaxis
        ax.set_yticks(yticks, labels=[str(yaxis_fun(states[y][0])) for y in yticks])
    colors = np.zeros((n_states, N, 4))
    max_prob = np.max(eigvecs[::2, :]**2 + eigvecs[1::2, :]**2)
    for n_state in range(n_states):
        max_prob = np.max(eigvecs[::2, n_state]**2 + eigvecs[1::2, n_state]**2)
        for i in range(N):
            colors[n_state, i] = _get_color(eigvecs[2*i, n_state], eigvecs[2*i+1, n_state], colored, max_prob=max_prob)
    ax.imshow(colors, aspect='auto', origin='lower')
    xticks = list(range(N//5-1, N, N//5))
    ax.set_xticks(xticks, labels=[str(x+1) for x in xticks])
    ax.set_xlabel(r"$n_{\text{cell}}$")
    ax.set_ylabel(ylabel)

    if show_legend:
        patch_A = Patch(color=color_A, label='only sublattice A')
        patch_B = Patch(color=color_B, label='only sublattice B')
        patch_anti = Patch(color=color_anti, label='A and B anti-symmetric')
        patch_sym = Patch(color=color_sym, label='A and B symmetric')
        patch_none = Patch(color=(1, 1, 1, 0), label='alpha value = probability in unit cell')
        plt.legend(handles=[patch_A, patch_B, patch_anti, patch_sym, patch_none], bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_localization(params: Iterable[ChainModel.Params] | Callable[[Harper.Frequency], Iterable[ChainModel.Params]],
                      xaxis: Callable[[ChainModel.Params], float] = xaxis_getter("harper.phi_H"),
                      xlabel: str = r"$\varphi_H / \frac{\pi}{4}$",
                      ax: plt.Axes | None = None,
                      modus: str = "mean",
                      methods: LocalizationMeasure.Method | Iterable[LocalizationMeasure.Method] \
                              = LocalizationMeasure.METHODS,
                      ylim: tuple[float, float] = (-.05, 1.05),
                      alphas: Sequence[Harper.Frequency] | None = None,
                      phase_jump: float | None = 1,
                      ylabel: str = "L",
                      show_legend: Literal["all", "methods", "lines", "methods_alphas"] | bool = False,
                      alphas_special: Sequence[Harper.Frequency] = tuple(),
                      show_theoretical: bool = True,
                      show_title: bool = True,
                      plot_num_every: int = 1,
                      ):

    alphas_unspecial: Sequence[Harper.Frequency]
    xlabel = xlabel or ( r"$\alpha$" if xaxis == "alpha" else xaxis if isinstance(xaxis, str) else "")
    if isinstance(params, Iterable):
        params = tuple(params)
        params_list = [params]
        if alphas_special:
            raise ValueError("Alphas_special must not be given if params is an iterable")
        if alphas is not None:
            raise ValueError("Alphas must not be given if params is an iterable")
        alpha = params[0].harper_not_none.alpha
        if not all(param.harper and param.harper.alpha == alpha for param in params):
            raise ValueError("All parameters must have the same N")
        alphas = [alpha]
    else:
        if alphas is None:
            raise ValueError("Alphas must be given if params is a function")
        params_getter = params
        if alphas_special:
            alphas = list(alphas)
            alphas_special = list(alphas_special)
            alphas_unspecial = [alpha for alpha in alphas if alpha not in alphas_special]
            alphas = list(set(alphas_unspecial + alphas_special))
        else:
            alphas_unspecial = alphas
        params_list = [tuple(params_getter(alpha)) for alpha in alphas]
        for params in params_list:
            for param in params:
                if param.harper is None:
                    raise ValueError("All parameters must have a Harper part")
    x = np.asarray([xaxis(param) for param in params_list[0]])
    x_num = x[::plot_num_every]
    N = params_list[0][0].N
    if not all(param.N == N for params in params_list for param in params):
        raise ValueError("All parameters must have the same N")
    if isinstance(methods, LocalizationMeasure.Method):
        methods = [methods]

    linestyle_theo = "--"
    linestyle_num = "."
    linestyle_special = ":"
    linewidth_theo = 1
    linewidth_num = .5
    linewidth_special = .5
    markers = "." if len(alphas) == 1 else "<>v^+x"  # ".d^p*v<>sdD+xo8hHPX"

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10)

    if phase_jump is not None:
        ax.plot([0, 1, 1, 2], [0, 0, 1, 1], linestyle=":", color="k", linewidth=.5)

    L_alpha: dict[float, dict[LocalizationMeasure.Method, FloatArray]] = {}
    for alpha, params in zip(alphas, params_list):
        L_alpha[alpha] = LocalizationMeasure.measure_systems_all_methods(params[::plot_num_every], modus=modus)

    for method in methods:
        measure = LocalizationMeasure(method)

        L_theo = np.array([
            measure.theoretical_value(param.harper_not_none.localization_length,
                                      N=param.N, strict=False)
            for param in params_list[0]
        ])

        L_num = {"min": np.ones_like(x_num), "max": np.zeros_like(x_num)}
        for alpha in alphas_unspecial:
            loc_val = L_alpha[alpha][method]
            L_num["min"] = np.minimum(L_num["min"], loc_val)
            L_num["max"] = np.maximum(L_num["max"], loc_val)
        L_num["opt"] = L_num["min"].copy()
        idx = x_num > (phase_jump or 1)
        L_num["opt"][idx] = L_num["max"][idx]

        color = get_color(method)
        if show_theoretical:
            ax.plot(x, L_theo, linestyle_theo, color=color, linewidth=linewidth_theo,
                    label=f"{method.instance.latex_name} theo", zorder=1)
        for a, alpha in enumerate(alphas):
            marker = markers[a % 16]
            linestyle = linestyle_special if alpha in alphas_special else linestyle_num
            linewidth = linewidth_special if alpha in alphas_special else linewidth_num
            ax.plot(x_num, L_alpha[alpha][method], linestyle, color=color, linewidth=linewidth, markersize=1.5 if len(alphas) == 1 else 5,
                    zorder=2, label=f"{method.instance.latex_name} num", marker=marker)
        if any(alphas_special):
            ax.fill_between(x, L_num["min"], L_num["max"], alpha=.1, color=color, linewidth=0)

    handles: list[Artist] = []
    labels: list[str] = []
    def add_to_legend(label: str, colors: Sized, linestyles: Sized | str, **kwargs):
        n = len(colors) if isinstance(linestyles, str) else len(linestyles)
        handles.append(LineCollection([[(0, 0)]] * n, colors=colors, linestyles=linestyles, **kwargs))
        labels.append(label)
    colors = [get_color(method) for method in methods]

    if show_legend in ("all", "methods"):
        linestyles = [linestyle_theo] * show_theoretical + [linestyle_num] + [linestyle_special] * bool(alphas_special)
        linewidths = [linewidth_theo] * show_theoretical + [linewidth_num] + [linewidth_special] * bool(alphas_special)
        for method in methods:
            add_to_legend(method.instance.latex_name, get_color(method), linestyles, linewidth=linewidths)
    if show_legend in ("all", "lines"):
        if alphas_special:
            add_to_legend("theory", colors, linestyle_theo, linewidth=linewidth_theo)
            add_to_legend("$\\alpha\\in A_0$", colors, linestyle_num, linewidth=linewidth_num)
            add_to_legend("$\\alpha\\in A_1$", colors, linestyle_special, linewidth=linewidth_special)
        else:
            add_to_legend("theoretical", colors, linestyle_theo, linewidth=linewidth_theo)
            add_to_legend("numerical", colors, linestyle_num, linewidth=linewidth_num)
    if show_legend == "methods_alphas":
        for method in methods:
            labels.append(method.instance.latex_name)
            handles.append(Line2D([0], [0], marker="s", color=get_color(method), linestyle=""))
        for a, alpha in enumerate(alphas):
            labels.append(get_label(alpha))
            handles.append(Line2D([0], [0], marker=markers[a], color="gray", linestyle=""))

    if show_legend == True:
        ax.legend(fontsize="small")
    elif show_legend:
        ax.legend(handles, labels, handler_map={LineCollection: HandlerLineCollectionShowAll()},
                  fontsize="small")


    ax.set_xlim(min(x), max(x))
    ax.set_ylim(ylim)
    N_text = f"$N = {N}$"
    if show_title:
        ax.set_title(N_text)
    # ax.text(0.1, 0.9, N_text, transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def plot_localization_2d(params: ParamsArray2D,
                         method: LocalizationMeasure.Method,
                         xaxis: str | Callable[[ChainModel.Params], float] = "alpha",
                         xlabel: str | None = None,
                         yaxis: str | Callable[[ChainModel.Params], float] = "alpha",
                         ylabel: str | None = None,
                         ax: plt.Axes | None = None,
                         modus: str = "mean",
                         cmap: str = CMAP,
                         vlim: tuple[float | None, float | None] = (0, 1)):
    xlabel = xlabel or ( r"$\alpha$" if xaxis == "alpha" else xaxis if isinstance(xaxis, str) else "")
    ylabel = ylabel or ( r"$\alpha$" if yaxis == "alpha" else yaxis if isinstance(yaxis, str) else "")
    assert params.ndim == 2
    N = params[0, 0].N
    if not all(param.N == N for param in params.flatten()):
        raise ValueError("All parameters must have the same N")
    xaxis_fun = (lambda param: getattr(param, xaxis)) if isinstance(xaxis, str) else xaxis
    x = np.asarray([xaxis_fun(param) for param in params[0, :]])
    yaxis_fun = (lambda param: getattr(param, yaxis)) if isinstance(yaxis, str) else yaxis
    y = np.asarray([yaxis_fun(param) for param in params[:, 0]])

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10)

    localization_measure = LocalizationMeasure(method=method)
    z = np.vectorize(lambda p: localization_measure.measure_system(p, modus))(params)
    label = f"{method.instance.latex_name} {modus}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    h = ax.pcolor(x, y, z, shading='auto', vmin=vlim[0], vmax=vlim[1], cmap=cmap, label=label)
    # show colormap
    plt.colorbar(h, ax=ax, label="localization measure")

    return x, y, z


def plot_edge_states_alpha(box: tuple[float, float, float, float],
                     axs_specs,
                     axs_states,
                     N: int,
                     phi_H: float,
                     phi_hop: float,
                     beta: float | None = None):

    F = Harper.IrrationalFrequency
    alpha_opacity = .8
    ms = .6
    d_alpha = .001

    params_kwargs = {"phi_H": phi_H, "phi_hop": phi_hop, "N": N}
    if beta is not None:
        params_kwargs["beta"] = beta

    for j, (factor, label) in enumerate(((None, "commensurable"),
                                         (F.GOLDEN_NUMBER, "incommensurable"))):

        ax = flatten_axes(axs_specs)[j]
        xlim = (.5 - box[1], .5 - box[0]) if factor is None else (.5 + box[0], .5 + box[1])
        alpha = sum(xlim) / len(xlim)

        if factor is None:
            params = ParamsRange("alpha", (xlim[0], xlim[1], d_alpha),
                                 alpha_getter_kwargs=None, **params_kwargs)
        elif isinstance(factor, Harper.IrrationalFrequency):
            params = ParamsRange("alpha", (
            d_alpha * (int(xlim[0] / d_alpha / factor) + 1), d_alpha * (int(xlim[1] / d_alpha / factor) - 1), d_alpha,
            factor), alpha_getter_kwargs=None, **params_kwargs)
        else:
            raise ValueError(factor)

        p = params[np.argmin(np.abs(params.param_values - alpha))]
        edge_measure = LocalizationMeasure(method=LocalizationMeasure.EDGE_METHOD)
        loc_vals = edge_measure.measure_all(p)
        max_indexes = loc_vals.argsort()[-2:][::-1]
        min_indexes = loc_vals.argsort()[:2]
        states_left = tuple((p, int(idx + 1)) for idx in min_indexes)
        states_right = tuple((p, int(idx + 1)) for idx in max_indexes)
        states = states_left + states_right

        plot_eigvals(params=params, ax=ax, text="", do_compute_bands=False,
                              xaxis=xaxis_getter("harper.alpha"), xlabel=r"$\alpha$",
                              line_style=" ", marker="o", markersize=ms, markerfacecolor="black",
                              markeredgewidth=0,
                              states=states, alpha=alpha_opacity, ylim=(-2, 2),
                              state_marks=["o", "o", "x", "x"],
                              state_color_idx=[_i + 2 * j for _i in (0, 1, 0, 1)])
        ax.set_xlim(*xlim)
        ax.set_ylim(*box[-2:])

        for i, states_left_right in enumerate(zip(states_left, states_right)):
            l = i + 2 * j
            for k, state in enumerate(states_left_right):
                ax = axs_states[l, k]
                plot_state(state, ax=ax, plot_probability=True, psi_index=str(state[1]),
                                    color_index=l, show_xlabel=l == 3, show_ylabel=True,
                                    show_legend=False)
                alpha = state[0].harper.alpha
                alpha_str = f"{alpha / (factor or 1):.3f}" + r"\alpha_\mathrm{G}$" + "\n$=" + str(alpha)[
                                                                                       :6] + r"\ldots" if label.startswith(
                    "in") else str(alpha)
                ax.text(.96, .9, r"$\alpha=" + alpha_str + r"$", ha="right", va="top", fontsize="x-small",
                        transform=ax.transAxes)
                NM = N if phi_hop == 1 else N // 2
                xlim = (.5, NM // 10 - .5) if k == 0 else (9 * NM // 10 + .5, NM + .5)
                xticks = (1, NM // 20) if k == 0 else (19 * NM // 20, NM)
                ax.set_xlim(*xlim)
                ax.set_xticks(xticks)