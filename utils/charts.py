import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = px.colors.qualitative.Vivid


def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
    series = df[column].dropna()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Distribution de {column}", f"Box plot - {column}"),
        column_widths=[0.6, 0.4],
    )

    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=30,
            name=column,
            marker_color=COLOR_PALETTE[0],
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Box(y=series, name=column, marker_color=COLOR_PALETTE[1], showlegend=False),
        row=1,
        col=2,
    )

    fig.update_layout(template=PLOTLY_TEMPLATE, height=450, showlegend=False)
    fig.update_xaxes(title_text=column, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600, title="Matrice de corrélation")
    return fig


def create_outlier_plot(df: pd.DataFrame, column: str) -> go.Figure:
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series[(series >= lower) & (series <= upper)].index,
            y=series[(series >= lower) & (series <= upper)],
            mode="markers",
            name="Normal",
            marker=dict(color=COLOR_PALETTE[0], opacity=0.5, size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=outliers.index,
            y=outliers,
            mode="markers",
            name=f"Outliers ({len(outliers)})",
            marker=dict(color="red", opacity=0.8, size=8, symbol="x"),
        )
    )
    fig.add_hline(
        y=upper,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Upper: {upper:.2f}",
    )
    fig.add_hline(
        y=lower,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Lower: {lower:.2f}",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        title=f"Outlier Detection - {column} ({len(outliers)} outliers détectés via IQR)",
        xaxis_title="Index",
        yaxis_title=column,
    )
    return fig


def create_trend_plot(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    plot_df = df[[date_col, value_col]].dropna().sort_values(date_col)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[date_col],
            y=plot_df[value_col],
            mode="lines+markers",
            name=value_col,
            line=dict(color=COLOR_PALETTE[0], width=2),
            marker=dict(size=4),
        )
    )

    rolling = (
        plot_df.set_index(date_col)[value_col]
        .rolling(window=min(7, len(plot_df)), min_periods=1)
        .mean()
    )
    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name="Moyenne mobile (7 périodes)",
            line=dict(color=COLOR_PALETTE[1], width=2, dash="dash"),
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        title=f"Tendance - {value_col} dans le temps",
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return fig


def create_group_comparison(df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Moyenne de {num_col} par {cat_col}",
            "Distribution par groupe",
        ),
        column_widths=[0.5, 0.5],
    )

    means = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(15)
    fig.add_trace(
        go.Bar(
            x=means.index.astype(str),
            y=means.values,
            marker_color=COLOR_PALETTE[0],
            name="Moyenne",
        ),
        row=1,
        col=1,
    )

    top_cats = df[cat_col].value_counts().head(5).index.tolist()
    box_data = df[df[cat_col].isin(top_cats)]
    for i, cat in enumerate(top_cats):
        subset = box_data[box_data[cat_col] == cat][num_col].dropna()
        fig.add_trace(
            go.Box(
                y=subset,
                name=str(cat),
                marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=False,
        title=f"Comparaison : {num_col} par {cat_col}",
    )
    return fig


def create_categorical_plot(df: pd.DataFrame, column: str) -> go.Figure:
    vc = df[column].value_counts().head(15)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Top valeurs - {column}", "Proportions"),
        specs=[[{"type": "bar"}, {"type": "pie"}]],
    )

    fig.add_trace(
        go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=COLOR_PALETTE[: len(vc)]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Pie(labels=vc.index.astype(str), values=vc.values, marker_colors=COLOR_PALETTE),
        row=1,
        col=2,
    )

    fig.update_layout(template=PLOTLY_TEMPLATE, height=450, showlegend=False)
    return fig


def create_scatter_plot(
    df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.6,
        title=f"{x_col} vs {y_col}" + (f" (coloré par {color_col})" if color_col else ""),
    )

    valid = df[[x_col, y_col]].dropna()
    slope, intercept, r_value, _, _ = scipy_stats.linregress(valid[x_col], valid[y_col])
    x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=slope * x_range + intercept,
            mode="lines",
            name=f"Trend (R²={r_value**2:.3f})",
            line=dict(color="yellow", dash="dash"),
        )
    )

    fig.update_layout(height=500)
    return fig
