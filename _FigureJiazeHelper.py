import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colour
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.sparse import eye, diags, issparse, csr_matrix


# color set
basic_line_color = '#808080'
black = '#222222'
darkgray = '#A9A9A9'
highlight = '#00B2EE'
gray = "#C0C0C0"
colors_red = ['#F01F1F', '#F385EE']
colors_blue = ["#130DF7", '#2D83E9', '#a6cee3', '#15D4B7']
colors_green = ['#13B632', '#13F121', '#BAEB34']
colors_yellow = ['#F0FA0D']


def gen_colors(start, end, n):
    start = colour.Color(start)
    end = colour.Color(end)
    colors = [c.hex_l for c in list(start.range_to(end, n))]
    return colors


def plot_block_matrix(reorder_A, partition_names=None, partition_counts=None, colors=None, ms=1,
                      save_path=None, label='Adjacency matrix', show_legend=True,
                      show_thislevel_partition=False, thislevel_partition_counts=None,
                      show_highlevel_partition=False, highlevel_partition_counts=None, color_by_row=False):
    if colors is None:
        colors = []
    if partition_counts is None:
        partition_counts = []
    if partition_names is None:
        partition_names = []
    plt.spy(reorder_A, markersize=ms, rasterized=True, color=gray)
    accumulate_count = 0
    for i in range(len(partition_counts)):
        cur_counts = int(partition_counts[i])
        accumulate_count += cur_counts
        temp_A = np.copy(reorder_A)
        temp_A[accumulate_count:, :] = 0
        if color_by_row is False:
            temp_A[:, accumulate_count:] = 0
        temp_A[:accumulate_count-cur_counts, :] = 0
        if color_by_row is False:
            temp_A[:, :accumulate_count-cur_counts] = 0
        plt.spy(temp_A, markersize=ms, rasterized=True, color=colors[i], label=partition_names[i])
    if show_thislevel_partition:
        counts_sum = 0
        for count in thislevel_partition_counts[:-1]:
            counts_sum += count
            plt.axvline(counts_sum, color='grey', lw=1)
            plt.axhline(counts_sum, color='grey', lw=1)
    if show_highlevel_partition:
        counts_sum = 0
        for count in highlevel_partition_counts[:-1]:
            counts_sum += count
            plt.axvline(counts_sum, color='k', lw=2)
            plt.axhline(counts_sum, color='k', lw=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(label)
    if show_legend:
        plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=10, markerscale=2/ms)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)


def color_scatter_2d(x, y, z, z_center, title, xlabel, ylabel, min_z=None, max_z=None, cmap=cm.coolwarm,
                     save_path=None, ax=None, fig=None):
    """
    scatter z for x,y-axis. Divided by z_center, z is colored different. all value in z > 0
    """
    C = [0] * np.shape(z)[0]
    minz = min_z if min_z is not None else np.min(z)
    maxz = max_z if max_z is not None else np.max(z)
    print(f'min={minz}, max={maxz}')
    for i in range(np.size(z)):
        if minz <= z[i] <= z_center:
            C[i] = cmap((z[i] - minz) / (z_center - minz) * 0.5)
        elif z[i] > z_center:
            C[i] = cmap((z[i] - z_center) / (maxz - z_center) * 0.5 + 0.5)
        else:
            pass
    if ax is None or fig is None:
        fig = plt.figure(figsize=(10, 10))
        widths = [4]
        heights = [4]
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths, height_ratios=heights)
        row = 0
        col = 0
        ax = fig.add_subplot(spec5[row, col])
    # plt.axhline(0, linestyle=':')
    p = ax.scatter(x, y, s=10, c=C)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, rotation='horizontal', fontsize=20)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    print(f'{np.min(y)}, {np.max(y)}')
    ax.set_xticks(np.arange(np.min(x), np.max(x)+0.1, 0.1))
    ax.set_yticks(np.linspace(np.min(y), np.max(y), np.size(np.unique(y))))
    # plt.axhline(-1/3, color='k', lw=1)
    # plt.axvline(0.5, linestyle=':')
    # ax.set_zlabel(r'$\frac{SNR_2}{SNR_3}$')
    cticks = [0, 0.25, 0.5, 0.75, 1]
    clabels = [str(np.round(2 * x * (z_center - minz) + minz, 3)) if x < 0.5 else str(
        np.round((2 * x - 1) * (maxz - z_center) + z_center, 3)) for x in cticks]
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ticks=cticks, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_yticklabels(clabels)
    # plt.legend(loc=1, fontsize=10, markerscale=4)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)


def color_imshow_2d(x, y, z, z_center, title="", xlabel="", ylabel="", min_z=None, max_z=None, cmap=cm.coolwarm, save_path=None,
                 ax=None, fig=None, return_gridz=False, set_xticks=True, set_yticks=True, vmin=None, vmax=None, ytickprecision=5, cticks=None, show_colorbar=True, show_ext_color=False):
    minz = min_z if min_z is not None else np.min(z)
    maxz = max_z if max_z is not None else np.max(z)
    norm_z = np.zeros(np.size(z))
    for i in range(np.size(z)):
        if np.abs(z[i]) < 1e-5:
            z[i] = 0
        if z[i] < minz:
            norm_z[i] = z[i]
        if z[i] > maxz:
            norm_z[i] = z[i]
        if minz <= z[i] <= z_center:
            norm_z[i] = (z[i] - minz) / (z_center - minz) * 0.5
        elif z[i] > z_center:
            norm_z[i] = (z[i] - z_center) / (maxz - z_center) * 0.5 + 0.5
        else:
            norm_z[i] = z[i] if z[i] < 0 else -1
    _x = np.sort(np.unique(x))
    # print(np.size(_x))
    _y = np.sort(np.unique(y))
    _z = np.zeros((np.size(_y), np.size(_x)))
    ori_z = np.zeros((np.size(_y), np.size(_x)))
    for i, zi in enumerate(norm_z):
        i_x = np.where(_x == x[i])
        i_y = np.where(_y == y[i])
        _z[i_y, i_x] = zi
        ori_z[i_y, i_x] = z[i]
    if return_gridz:
        return ori_z
    if ax is None or fig is None:
        fig = plt.figure(figsize=(10, 10))
        widths = [4]
        heights = [4]
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths, height_ratios=heights)
        row = 0
        col = 0
        ax = fig.add_subplot(spec5[row, col])
    if vmax is not None:
        norm_vmax = ((vmax - minz) / (z_center - minz) * 0.5) if vmax <= z_center else ((vmax - z_center) / (maxz - z_center) * 0.5 + 0.5)
    else:
        norm_vmax = None
    if vmin is not None:
        norm_vmin = ((vmin - minz) / (z_center - minz) * 0.5) if vmin <= z_center else ((vmin - z_center) / (maxz - z_center) * 0.5 + 0.5)
    else:
        norm_vmin = None
    im = ax.imshow(_z, cmap=cmap, origin='lower', vmin=norm_vmin, vmax=norm_vmax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, rotation='horizontal', fontsize=16)
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(y), np.max(y))
    # print(f'{np.min(y)}, {np.max(y)}')
    # smft = plt.ScalarFormatter(useMathText=True)
    # smft.set_powerlimits((-2, 2))
    # smft.set_useMathText(True)
    # ax.xaxis.set_major_formatter(smft)
    # ax.yaxis.set_major_formatter(smft)
    # ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
    # ax.yaxis.get_major_formatter().set_useOffset(True)
    # ax.yaxis.get_major_formatter().set_useMathText(True)
    if set_xticks:
        ax.set_xticks(np.linspace(0, np.size(_x)-1, 3), np.around(np.linspace(np.min(x), np.max(x), 3), 3), fontsize=10)
    if set_yticks:
        yticks_num = 5  # int(np.size(np.unique(y)) / 2)
        ax.set_yticks(np.linspace(0, np.size(_y) - 1, yticks_num), np.around(np.linspace(np.min(y), np.max(y), yticks_num), ytickprecision), fontsize=10)
    else:
        ax.set_yticks([])
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # cticks = [0, 0.25, 0.5, 0.75, 1]
        # cticks = [0, 0.5, 1 if vmax is None else norm_vmax
        if cticks is None:
            if vmax is None and maxz <= 1:
                cticks = np.linspace(0, 1, 2)
            elif vmax is None:
                cticks = np.linspace(0, 1, int(maxz))
            else:
                cticks = np.linspace(0, norm_vmax, int(vmax))
        # cticks = np.linspace(0, 1 if vmax is None else norm_vmax, 2 if vmax is None else int(vmax))
        clabels = [str(int(np.round(np.abs(2 * x * (z_center - minz) + minz), 0))) if x < 0.5 else str(
            int(np.round((2 * x - 1) * (maxz - z_center) + z_center, 0))) for x in cticks]
        print(f'minz={minz}, maxz={maxz}, cticks={cticks}, clabel={clabels}')
        if show_ext_color:
            cbar = fig.colorbar(im, cax=cax, ticks=cticks, extend='both')
        else:
            cbar = fig.colorbar(im, cax=cax, ticks=cticks)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_yticklabels(clabels)
        cbar.outline.set(visible=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    return ori_z


def contour_data(data, ax, levels, fmt, color='white', linestyle=None, inline=True, printrange=False):
    CS = ax.contour(data, levels=levels, colors=color, linewidths=1.2, linestyles=linestyle)
    if inline:
        ax.clabel(CS, fmt=fmt, inline=1, fontsize=9)
    if printrange:
        print(f'min={np.min(data)}, max={np.max(data)}')
    handles, labels = CS.legend_elements()
    return handles


def get_confusionmatrix(truePartition, cdPartition, trueNumgroup, cdNumgroup):
    confusionMatrix = np.zeros((trueNumgroup, cdNumgroup))
    uniqueTpartition = np.unique(truePartition)
    uniqueDpartition = np.unique(cdPartition)
    for iTrue in uniqueTpartition:
        trueIndex = np.where(truePartition == iTrue)
        for iCD in uniqueDpartition:
            i = np.where(uniqueTpartition == iTrue)
            j = np.where(uniqueDpartition == iCD)
            confusionMatrix[i, j] = np.size(np.where(cdPartition[trueIndex]==iCD))
#     print(confusionMatrix)
    true_ind, CD_ind = linear_sum_assignment(confusionMatrix, maximize=True)
    print(f'True index is {true_ind}, Community detected index is {CD_ind}')
    confusionMatrix[:, np.sort(CD_ind)] = confusionMatrix[:, CD_ind]
    return confusionMatrix, CD_ind


def reorder_A_by_partition(A, partition, need_array=True):
    """ partition: vec from 0 to numgroup-1 """
    pvec = partition
    reorder_index = np.argsort(pvec)
    if need_array:
        if issparse(A):
            reordered_A = A.toarray()
        else:
            reordered_A = np.copy(A)
        # print(A)
        reordered_A = reordered_A[reorder_index, :][:, reorder_index]
    else:
        reordered_A = None
    partitions, counts = np.unique(pvec, return_counts=True)
    partition_counts = np.zeros(np.size(counts))
    for i in range(np.size(counts)):
        partition_counts[partitions[i]] = counts[i]
    partition_names = list(range(np.size(partition_counts)))
    return reordered_A, reorder_index, partition_names, partition_counts


def reorder_inner_block(reorder_A_0, reorder_i_0, partition_names_0, partition_counts_0, reorder_i_1, partition_names_1,
                        partition_counts_1):
    reorder_index = None
    reorder_partition_names_1 = []
    reorder_partition_counts_1 = np.array([])
    for p0, p0_i in zip(partition_counts_0, list(range(len(partition_counts_0)))):
        # find current block
        current_block_begin = int(np.sum(partition_counts_0[:p0_i]))
        current_block_end = int(np.sum(partition_counts_0[:p0_i + 1]))
        # print(current_block_begin, current_block_end)
        current_block_i0 = reorder_i_0[current_block_begin:current_block_end]
        # find partition 1 in this block
        current_block_part1 = np.zeros(np.size(current_block_i0))
        for i, real_i in enumerate(current_block_i0):
            i1_index = np.where(reorder_i_1 == real_i)[0][0]
            part1 = None
            for p1 in range(len(partition_counts_1)):
                if np.sum(partition_counts_1[:p1]) <= i1_index < np.sum(partition_counts_1[:p1 + 1]):
                    part1 = p1
                    break
            current_block_part1[i] = part1
        # reorder this block
        curren_block_reorder_index = np.argsort(current_block_part1)
        # splice the reorder index
        curren_block_reorder_index += current_block_begin
        if reorder_index is None:
            reorder_index = curren_block_reorder_index
        else:
            reorder_index = np.hstack((reorder_index, curren_block_reorder_index)).astype('int64')
        # get reordered partition 1 counts and names
        partitions, counts = np.unique(current_block_part1, return_counts=True)
        for i in range(len(counts)):
            reorder_partition_names_1.append(partition_names_1[int(partitions[i])])
            reorder_partition_counts_1 = np.hstack((reorder_partition_counts_1, counts[i])).astype('int64')

    reorder_A = np.copy(reorder_A_0)
    reorder_A = reorder_A[reorder_index, :][:, reorder_index]
    reorder_i = np.copy(reorder_i_0)
    reorder_i = reorder_i[reorder_index]
    return reorder_A, reorder_i, partition_names_0, partition_counts_0, reorder_partition_names_1, reorder_partition_counts_1


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_cm(confusionMatrix, fig=None, ax=None):
    subTrueNumgroup,subBHNumgroup = np.shape(confusionMatrix)
    rowsum = np.sum(confusionMatrix, axis=1)
    rowsum = rowsum.reshape(-1, 1)
    rowsum = np.repeat(rowsum, subBHNumgroup, axis=1)
    normConfusionMatrix = np.round(confusionMatrix / rowsum, 2)
    if fig is None and ax is None:
        fig = plt.figure(figsize=(3, 3))
        widths = [4]
        heights = [4]
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths, height_ratios=heights)
        row = 0
        col = 0
        ax = fig.add_subplot(spec5[row, col])
#     cmap = mpl.colormaps["bwr_r"]
#     cmap = mpl.colormaps["RdYlBu"]
    cmap = mpl.colormaps["seismic_r"]
    cmap = truncate_colormap(cmap, 0.3, 0.7)
    im = ax.matshow(normConfusionMatrix, cmap=cmap, vmin=0, vmax=1)
    for i in range(subTrueNumgroup):
        for j in range(subBHNumgroup):
            c = normConfusionMatrix[i,j]
            ax.text(j, i, str(c), va='center', ha='center', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    return im