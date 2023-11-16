from collections import OrderedDict

from plotly import optional_imports
from plotly.graph_objs import graph_objs
from copy import deepcopy
from tqdm import tqdm

from . import linkage_matrix_funcs
from .textual_boxplot import textual_boxplot

# Optional imports, may be None for users that only use our core functionality.
np = optional_imports.get_module("numpy")
scp = optional_imports.get_module("scipy")
sch = optional_imports.get_module("scipy.cluster.hierarchy")
scs = optional_imports.get_module("scipy.spatial")

def generate_graph_dendrogram(hierarchy, G, visual_filepath='dendrogram.html', p=5, color_threshold=0.5, width=800, height=800, title=None, x_axis_label="Cluster Size", show=True, cluster_labels_dict=None):
    """
    Generates a dendrogram visualisation for the given graph and its hierarchy.

    The dendrogram is saved to an HTML file and can also be displayed in a web browser.

    :param hierarchy: A hierarchy of partitions of the graph (list of tuples of node sets).
    :param G: The graph from which the hierarchy is derived.
    :param visual_filepath: The filepath for the HTML output (default 'dendrogram.html').
    :param p: The number of steps to show in the truncated dendrogram (default 5).
    :param color_threshold: The threshold to color clusters in the dendrogram (default 0.5).
    :param width: The width of the rendered dendrogram in pixels (default 800).
    :param height: The height of the rendered dendrogram in pixels (default 800).
    :param show: If True, display the dendrogram in the default web browser (default True).

    :return: None. Outputs an HTML file at the specified path and optionally displays the dendrogram.

    The function starts by generating a linkage matrix from the hierarchy and the graph. It then creates a dendrogram
    using this linkage matrix, applying the specified color threshold and dimensions. Additional metadata, such as
    the most connected nodes and date information, are calculated and formatted for inclusion as hover-over text in
    the dendrogram. The visualisation is saved as an HTML file and can be displayed in the browser.

    Note: Placeholder values are used for missing 'date_added' information in the graph's node attributes. This
    should be updated with actual data for the complete network.
    """

    # Generate linkage matrix
    Z, set_labels = linkage_matrix_funcs.generate_linkage_matrix(hierarchy, G)

    # Flip the labels for easy lookup
    flipped_labels = {v: k for k, v in set_labels.items()}

    def get_most_connected_nodes(G, N=3):
        node_degrees = list(G.degree())
        sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)
        return sorted_nodes[:N]

    def get_date_added(G):
        dates = []
        for node in G:
            try:
                dates.append(G.nodes[node]['date_added'])
            except:
                dates.append(1688169600) #TODO: Remove. I put this in as a placeholder because I don't have everyone's add date since the data was pulled over the summer. Timestamp is July 1st 2023
                continue
        return dates

    import datetime

    def timestamp_to_float(ts):
        # Convert the timestamp to a date
        dt = datetime.datetime.utcfromtimestamp(ts)
        year = dt.year

        # Determine the start and end of the year in timestamp format
        start_of_year = datetime.datetime(year, 1, 1)
        end_of_year = datetime.datetime(year + 1, 1, 1)
        start_of_year_ts = int(start_of_year.timestamp())
        end_of_year_ts = int(end_of_year.timestamp())

        # Calculate the fraction of the year that has passed
        fraction_passed = (ts - start_of_year_ts) / (end_of_year_ts - start_of_year_ts)

        # Return the formatted float
        return year + fraction_passed

    # Generate label mapping
    label_mapping = {}
    for k, v in flipped_labels.items():
        H = G.subgraph([node for node in v])
        label_mapping[str(k)] =  f"{', '.join([str(node[0])+' ('+str(node[1])+')' for node in get_most_connected_nodes(H, N=3)])} |\t {len(H.nodes):<3}"


    # Generate hover text
    hovertext = []
    graph_year_times = [timestamp_to_float(ts) for ts in get_date_added(G)]

    for elem in tqdm(linkage_matrix_funcs.pruned_linkage_matrix_to_list(deepcopy(Z), p=p)):
        H = G.subgraph(list(flipped_labels[elem])).copy()  # Create subgraph based on cluster
        text = ""
        if cluster_labels_dict and (elem in cluster_labels_dict):
            text += f"<b>{cluster_labels_dict[elem]}</b></br>"
        text += f"<b>Cluster:</b>: {elem}<br>"
        text += f"<b># of Users</b>: {len(H.nodes)}<br>"
        # text += f"<b>Modularity</b>: {}"
        text += "<b>Most Connected In Cluster:</b> <br>"
        for user in get_most_connected_nodes(H, 10):
            text += f"{user[0]}:\t\t{user[1]}<br>"
        text += f"<br><b>Dates Added in Cluster</b><br>"

        subgraph_year_times = [timestamp_to_float(ts) for ts in get_date_added(H)]

        # print(subgraph_year_times)
        text += textual_boxplot(
            subgraph_year_times,
            display_min= int(min(graph_year_times) // 1),
            display_max= int((max(graph_year_times) // 1) + 1),
            total_length=30
        )
        hovertext.append(text)

    # Generate dendrogram
    fig = create_z_dendrogram(linkage_matrix_funcs.generate_linkage_matrix(hierarchy, G)[0], orientation='left', color_threshold=color_threshold, truncate_mode='level', p=p, hovertext=hovertext, label_mapping=label_mapping)

    # Customize the layout
    fig.update_layout(width=width, height=height, hoverlabel=dict(
        font_family="Courier New, monospace",
        font_size=16
    ))

    if title:
        fig.update_layout(title=dict(text=title))

    fig.update_xaxes(title_text=x_axis_label)

    # Save to HTML and display
    fig.write_html(visual_filepath)
    if show:
        fig.show()




def create_z_dendrogram(
        Z,
        orientation="bottom",
        labels=None,
        colorscale=None,
        hovertext=None,
        color_threshold=None,
        truncate_mode=None,
        p=None,
        label_mapping=None
):
    """
    Function that returns a dendrogram Plotly figure object. This function
    utilizes a pre-computed linkage matrix 'Z' and serves as a thin
    wrapper around scipy.cluster.hierarchy.dendrogram.

    See also https://dash.plot.ly/dash-bio/clustergram.

    :param (ndarray) Z: Linkage matrix representing hierarchical clustering.
                        Each row represents a single linkage operation
                        with format [idx1, idx2, distance, sample_count].
    :param (str) orientation: 'top', 'right', 'bottom', or 'left'.
    :param (list) labels: List of axis category labels (observation labels).
    :param (list) colorscale: Optional colorscale for the dendrogram tree.
                              Requires 8 colors to be specified, the 7th of
                              which is ignored. With scipy>=1.5.0, the 2nd, 3rd,
                              and 6th are used twice as often as the others.
                              Given a shorter list, the missing values are
                              replaced with defaults and with a longer list the
                              extra values are ignored.
    :param (list[list]) hovertext: List of hovertext for constituent traces of dendrogram
                                  clusters.
    :param (double) color_threshold: Value at which the separation of clusters will be made.
    :param (str) truncate_mode: Determines how the dendrogram tree is truncated.
                                Supported modes: 'level', 'lastp', and 'mlab'.
                                Default is 'level'.
    :param (int or str) p: For 'lastp' truncate mode, specifies the number of
                          last merged clusters to be shown. For 'level' mode,
                          specifies the level at which the dendrogram is
                          truncated. Default is '10'.
    """
    if not scp or not scs or not sch:
        raise ImportError(
            "FigureFactory.create_dendrogram requires scipy, \
                            scipy.spatial and scipy.hierarchy"
        )



    dendrogram = _ZDendrogram(
        Z,
        orientation,
        labels,
        colorscale,
        hovertext=hovertext,
        color_threshold=color_threshold,
        truncate_mode=truncate_mode,
        p=p,
        label_mapping=label_mapping
    )

    return graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)


class _ZDendrogram(object):
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(
            self,
            Z,
            orientation="bottom",
            labels=None,
            colorscale=None,
            width=np.inf,
            height=np.inf,
            xaxis="xaxis",
            yaxis="yaxis",
            hovertext=None,
            color_threshold=None,
            truncate_mode=None,
            p=None,
            label_mapping=None
    ):
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}

        # New additions
        self.truncate_mode = truncate_mode
        self.p = p

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1


        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_dendrogram_traces(
            Z, colorscale, hovertext, color_threshold, truncate_mode, p
        )

        if label_mapping:
            self.labels = [label_mapping[label] for label in ordered_labels]
        else:
            self.labels = ordered_labels

        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        """
        Returns colorscale used for dendrogram tree clusters.

        :param (list) colorscale: Colors to use for the plot in rgb format.
        :rtype (dict): A dict of default colors mapped to the user colorscale.

        """

        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        # This list is the colors that can be used by dendrogram, which were
        # determined as the combination of the default above_threshold_color and
        # the default color palette (see scipy/cluster/hierarchy.py)
        d = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            # TODO: 'w' doesn't seem to be in the default color
            # palette in scipy/cluster/hierarchy.py
            "w": "white",
        }
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            rgb_colorscale = [
                "rgb(0,116,217)",  # blue
                "rgb(35,205,205)",  # cyan
                "rgb(61,153,112)",  # green
                "rgb(40,35,35)",  # black
                "rgb(133,20,75)",  # magenta
                "rgb(255,65,54)",  # red
                "rgb(255,255,255)",  # white
                "rgb(255,220,0)",  # yellow
            ]
        else:
            rgb_colorscale = colorscale

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]

        # add support for cyclic format colors as introduced in scipy===1.5.0
        # before this, the colors were named 'r', 'b', 'y' etc., now they are
        # named 'C0', 'C1', etc. To keep the colors consistent regardless of the
        # scipy version, we try as much as possible to map the new colors to the
        # old colors
        # this mapping was found by inpecting scipy/cluster/hierarchy.py (see
        # comment above).
        new_old_color_map = [
            ("C0", "b"),
            ("C1", "g"),
            ("C2", "r"),
            ("C3", "c"),
            ("C4", "m"),
            ("C5", "y"),
            ("C6", "k"),
            ("C7", "g"),
            ("C8", "r"),
            ("C9", "c"),
        ]
        for nc, oc in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                # it could happen that the old color isn't found (if a custom
                # colorscale was specified), in this case we set it to an
                # arbitrary default.
                default_colors[n] = "rgb(0,116,217)"

        return default_colors

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": True,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for dendrogram figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_dendrogram_traces(
            self, Z, colorscale, hovertext, color_threshold, truncate_mode, p
    ):
        """
        Computes elements required for plotting a dendrogram using a pre-computed linkage matrix 'Z'.

        :param (ndarray) Z: Linkage matrix representing hierarchical clustering.
                            Each row has the format [idx1, idx2, distance, sample_count].
        :param (list) colorscale: Color scale for dendrogram tree clusters.
        :param (list) hovertext: Hovertext for each trace of the dendrogram.
        :param (float) color_threshold: Threshold for coloring the dendrogram tree clusters.
        :param (str) truncate_mode: Mode for dendrogram truncation. Supported modes: 'level', 'lastp', and 'mlab'.
        :param (int or str) p: For 'lastp' truncate mode, specifies the number of
                              last merged clusters to be shown. For 'level' mode,
                              specifies the level at which the dendrogram is
                              truncated.

        :rtype (tuple): Returns tuple containing:
            (a) trace_list: List of Plotly trace objects for the dendrogram tree.
            (b) icoord: X coordinates of the dendrogram tree as arrays of length 4.
            (c) dcoord: Y coordinates of the dendrogram tree as arrays of length 4.
            (d) ordered_labels: Leaf labels in the order they appear on the plot.
            (e) P['leaves']: Left-to-right traversal of the leaves.
        """
        P = sch.dendrogram(
            Z,
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
            color_threshold=color_threshold,
            truncate_mode=truncate_mode,
            p=p
        )

        icoord = scp.array(P["icoord"])
        dcoord = scp.array(P["dcoord"])
        ordered_labels = scp.array(P["ivl"])
        color_list = scp.array(P["color_list"])
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=colors[color_key]),
                text=hovertext_label,
                hoverinfo="text",
            )

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, P["leaves"]