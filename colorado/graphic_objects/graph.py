# utilisies for graphs (.arg files)
from .bucket import get_aims_bucket_g_o
from plotly import graph_objects as go


def _list_buckets(graph, bucket_type, key_name=None):
    pass


def get_buckets_g_o_from_aims_graph(graph, bucket_names):
    """Get plotly scattersplots for the buckets listed in bucket_names

    Args:
        graph (aims graph): the graphs containing the buckets to plot
        bucket_names (Sequence[str]): A list of bucket names

    Returns:
        plotly.graphic_objects: a scatterplot graphic object
    """

    # call the function that returns a dictionnary of buckets named by key_name

    # call get_aims_bucket_g_o

    return None


def draw_buckets_from_aims_graph(graph, bucket_names, fig=None):
    """Draw buckets from a graph object

    Args:
        graph (aims graph): the graphs containing the buckets to plot
        bucket_names (Sequence[str]): A list of bucket names
        fig (plotly figure, optional): The figure where the plot happens. Defaults to None.

    Returns:
        [type]: [description]
    """
    if fig is None:
        fig = go.Figure()

    trace = get_buckets_g_o_from_aims_graph(graph, bucket_names)
    fig.add_trace(trace)

    return fig
