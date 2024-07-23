import plotly.graph_objects as go
import g2o


def plot_slam2d(optimizer, title):
    def edges_coord(edges, dim):
        for e in edges:
            yield e.vertices()[0].estimate()[dim]
            yield e.vertices()[1].estimate()[dim]
            yield None

    def edges_label(edges):
        for e in edges:
            yield f"{e.vertices()[0].id()}->{e.vertices()[1].id()} ({e.measurement().to_vector()})"
            yield ""
            yield None

    def print_edge(edge):
        return f"{edge.vertices()[0].id()}->{edge.vertices()[1].id()} ({edge.measurement().to_vector()})\n"

    fig = go.Figure()

    # edges
    edges = optimizer.edges()  # get set once to have same order
    se2_edges = [e for e in edges if type(e) == g2o.EdgeSE2]
    se2_edges_odom = [e for e in se2_edges if e.vertices()[0].id() == e.vertices()[1].id() - 1]
    se2_edges_loop = [e for e in se2_edges if e.vertices()[0].id() > e.vertices()[1].id()]
    se2_pointxy_edges = [e for e in edges if type(e) == g2o.EdgeSE2PointXY]

    print(f"Edges: {len(edges)}\n"
            f"SE2: {len(se2_edges)}\n"
            f"SE2 odometry: {len(se2_edges_odom)}\n"
            f"SE2 loop closure: {len(se2_edges_loop)}\n"
            f"SE2 PointXY: {len(se2_pointxy_edges)}")

    fig.add_trace(
        go.Scatter(
            x=list(edges_coord(se2_pointxy_edges, 0)),
            y=list(edges_coord(se2_pointxy_edges, 1)),
            mode="lines",
            line=dict(
                color='firebrick',
                width=2,
                dash='dash'),
            name="Measurement edges",
            legendgroup="Measurements"

        )
    )
    print("Added SE2 PointXY edges")

    fig.add_trace(
        go.Scatter(
            x=list(edges_coord(se2_edges_odom, 0)),
            y=list(edges_coord(se2_edges_odom, 1)),
            mode="lines",
            line=dict(
                color='midnightblue',
                width=4),
            name="Odometry pose edges",
            legendgroup="Odometry"
        )
    )
    print("Added SE2 odometry edges")

    fig.add_trace(
        go.Scatter(
            x=list(edges_coord(se2_edges_loop, 0)),
            y=list(edges_coord(se2_edges_loop, 1)),
            text=list(edges_label(se2_edges_loop)),
            mode="lines",
            line=dict(
                color='yellow',
                width=4),
            name="Loop closure pose edges",
            legendgroup="Weights"
        )
    )
    print("Added SE2 loop closure edges")

    # poses of the vertices
    vertices = optimizer.vertices()
    poses = [v for v in vertices.values() if type(v) == g2o.VertexSE2]
    measurements = [v.estimate() for v in vertices.values() if type(v) == g2o.VertexPointXY]

    fig.add_trace(
        go.Scatter(
            x=[v.estimate()[0] for v in poses],
            y=[v.estimate()[1] for v in poses],
            text=[f"Id: {v.id()}\n"
                  f"Angle: {v.estimate()[2]}\n"
                  f"Odometry:{[print_edge(edge) for edge in se2_edges_odom if edge.vertices()[0].id() == v.id()]}\n"
                  # f"Loop closure:{[print_edge(edge) for edge in se2_edges_loop if edge.vertices()[0].id() == v.id()]}\n"
                  for v in poses],
            mode="markers",
            marker_line_color="midnightblue",
            marker_color="lightskyblue",
            marker_line_width=2, marker_size=15,
            name="Poses",
            legendgroup="Poses"
        )
    )
    print("Added SE2 poses")

    fig.add_trace(
        go.Scatter(
            x=[v[0] for v in measurements],
            y=[v[1] for v in measurements],
            mode="markers",
            marker_symbol="star",
            marker_line_color="firebrick",
            marker_color="firebrick",
            marker_line_width=2, marker_size=15,
            name="Measurements",
            legendgroup="Measurements"
        )
    )
    print("Added PointXY measurements")

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(go.Layout({"title": title}))
    fig.update_yaxes(autorange="reversed")

    return fig
