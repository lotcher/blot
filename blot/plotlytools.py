import pandas as pd
import time
import copy
import numpy as np
# from plotly.graph_objs import *
from plotly.graph_objs import Figure, Layout, Bar, Box, Scatter, FigureWidget, Scatter3d, Histogram, Heatmap, Surface, \
    Pie
import plotly.figure_factory as ff
from collections import defaultdict
from IPython.display import display, Image
from .exceptions import BlotError
from .colors import normalize, get_scales, colorgen, to_rgba, get_colorscale
from .utils import check_kwargs, deep_update, kwargs_from_keyword, is_list
from . import tools
from . import offline
from . import auth
from . import ta

__TA_KWARGS = ['min_period', 'center', 'freq', 'how', 'rsi_upper', 'rsi_lower', 'boll_std', 'fast_period',
               'slow_period', 'signal_period', 'initial', 'af', 'open', 'high', 'low', 'close']


def iplot_to_dict(data):
    d = defaultdict(dict)
    for i in data:
        for k, v in list(i.items()):
            d[i['name']][k] = v
    return d


def dict_to_iplot(d):
    l = []
    for v in list(d.values()):
        l.append(v)
    return l


def _to_iplot(self, colors=None, colorscale=None, kind='scatter', mode='lines', interpolation='linear', symbol='dot',
              size='12', fill=None, fillcolor=None,
              width=3, dash='solid', sortbars=False, keys=None, bestfit=False, bestfit_colors=None, opacity=0.6,
              asDates=False, asTimestamp=False, text=None, **kwargs):
    """
    Generates a plotly Data object

    Parameters
    ----------
        colors : list or dict
            {key:color} to specify the color for each column
            [colors] to use the colors in the defined order
        colorscale : str
            Color scale name
            Only valid if 'colors' is null
            See blot.colors.scales() for available scales
        kind : string
            Kind of chart
                scatter
                bar
        mode : string
            Plotting mode for scatter trace
                lines
                markers
                lines+markers
                lines+text
                markers+text
                lines+markers+text
        interpolation : string
            Positioning of the connecting lines
                linear
                spline
                vhv
                hvh
                vh
                hv
        symbol : string
            The symbol that is drawn on the plot for each marker
            Valid only when mode includes markers
                circle
                circle-dot
                diamond
                square
                and many more...(see plotly.validators.scatter.marker.SymbolValidator.values)
        size : string or int
            Size of marker
            Valid only if marker in mode
        fill : bool
            Filled Traces
        width : int
            Line width
        dash : string
            Drawing style of lines
                solid
                dash
                dashdot
                dot
        sortbars : boole
            Sort bars in descending order
            * Only valid when kind='bar'
        keys : list of columns
            List of columns to chart.
            Also can be used for custom sorting.
        bestfit : boolean or list
            If True then a best fit line will be generated for
            all columns.
            If list then a best fit line will be generated for
            each key on the list.
        bestfit_colors : list or dict
            {key:color} to specify the color for each column
            [colors] to use the colors in the defined order
        asDates : bool
            If true it forces truncates times from a DatetimeIndex

    """
    df = self.copy()

    if asTimestamp:
        x = [_ for _ in df.index]
    elif df.index.__class__.__name__ in ('PeriodIndex', 'DatetimeIndex'):
        if asDates:
            df.index = df.index.date
        x = df.index.format()
    elif isinstance(df.index, pd.MultiIndex):
        x = ['({0})'.format(','.join([str(__) for __ in _])) for _ in df.index.values]
    else:
        x = df.index.values
    lines = {}
    if type(df) == pd.core.series.Series:
        df = pd.DataFrame({df.name: df})

    if not keys:
        if 'bar' in kind:
            if sortbars:
                keys = list(df.sum().sort_values(ascending=False).keys())
            else:
                keys = list(df.keys())
        else:
            keys = list(df.keys())

    colors = get_colors(colors, colorscale, keys)
    dash = get_items_as_list(dash, keys, 'dash')
    symbol = get_items_as_list(symbol, keys, 'symbol')
    mode = get_items_as_list(mode, keys, 'mode')
    interpolation = get_items_as_list(interpolation, keys, 'interpolation')
    width = get_items_as_list(width, keys, 'width')
    for i, key in enumerate(keys):
        lines[key] = {}
        lines[key]["x"] = x
        lines[key]["y"] = df[key].fillna('').values
        lines[key]["name"] = str(key)
        if text is not None:
            lines[key]["text"] = text
        if 'bar' in kind:
            lines[key]["marker"] = {'color': to_rgba(colors[key], opacity), 'line': {'color': colors[key], 'width': 1}}
        else:
            lines[key]["line"] = {'color': colors[key], 'width': width[key], 'dash': dash[key],
                                  'shape': interpolation[key]}
            lines[key]["mode"] = mode[key]
            if 'marker' in mode[key]:
                lines[key]["marker"] = dict(symbol=symbol[key], size=size)
            if fill:
                def current_fill():
                    if isinstance(fill, bool):
                        return 'tonexty' if kind == 'area' else 'tozeroy'
                    elif isinstance(fill, list):
                        return fill[i]
                    elif isinstance(fill, dict):
                        return fill.get(key, 'tonexty')
                    else:
                        return fill

                def current_fillcolor():
                    default = to_rgba(colors[key], kwargs.get('opacity', 0.3))
                    if isinstance(fillcolor, str):
                        return fillcolor
                    elif isinstance(fillcolor, list):
                        return fillcolor[i]
                    elif isinstance(fillcolor, dict):
                        return fillcolor.get(key, default)
                    else:
                        return default

                lines[key]["fill"] = current_fill()
                lines[key]["fillcolor"] = current_fillcolor()
    if 'bar' in kind:
        lines_plotly = [Bar(lines[key]).to_plotly_json() for key in keys]
    else:
        lines_plotly = [Scatter(lines[key]).to_plotly_json() for key in keys]
    for trace in lines_plotly:
        if isinstance(trace['name'], pd.Timestamp):
            trace.update(name=str(trace['name']))

    if bestfit:
        if isinstance(df.index, pd.MultiIndex):
            raise TypeError('x cannot be empty for MultiIndex dataframes')

        if type(bestfit) == list:
            keys = bestfit
        d = {}
        for key in keys:
            bestfit = df[key].bestfit()
            d[bestfit.formula] = bestfit
        bestfit_lines = pd.DataFrame(d).to_iplot(bestfit=False, colors=bestfit_colors, kind='scatter',
                                                 asTimestamp=asTimestamp)
        for line in bestfit_lines:
            line['line']['dash'] = 'dash'
            if not bestfit_colors:
                line['line']['color'] = to_rgba(line['line']['color'], .6)
        data = lines_plotly
        data.extend(bestfit_lines)
        return data

    return lines_plotly


def _iplot(
        self, kind='line', data=None, layout=None,

        x='', y='', y2='', z='', labels=None, values=None, text='', keys=None,

        title='', xtitle='', ytitle='', y2title='', ztitle='',

        theme=None, colors=None, colorscale=None, fill=None, fillcolor=None, width=None,
        dash='solid', mode='', interpolation='linear', symbol='circle', size=12,

        barmode='', bargap=None, bargroupgap=None, bins=None, histnorm='', boxpoints=False,

        orientation='v', annotations=None, vlines=None, hlines=None,

        gridcolor=None, zerolinecolor=None, margin=None, dimensions=None, subplots=False, as_figure=True, **kwargs
):
    """
    默认返回plotly Figure对象
    Parameters:
    -----------
        kind : string
            Kind of chart
                line：折线图
                scatter：散点图
                scatter3d：3d散点图
                bar：柱状图
                histogram：直方图
                box：箱形图
                spread：折线图对比，自动增加fig1-fig2子图
                ratio：折线图对比，自动增加fig1/fig2子图
                heatmap：热力图
                surface：曲面图
                bubble：气泡图
                bubble3d：3d气泡图
                pie：扇形图

        data : Data
            Plotly Data 对象.
            如果未显式指定，则自动从DataFrame对象生成

        layout : Layout
            Plotly layout 对象
            如果未显式指定，则自动从DataFrame对象生成

        ------------------------------------

        x : string
            代表x轴取值的列名

        y : string or list
            代表y轴取值的列名

        y2 : string or list(string)
            代表第二y轴取值的列名，用法同y相同，也可以和y同时使用。

        z : string
            代表z轴取值的列名

        labels : string or list
                string：单列作为label
                list：多列的组合作为label
            Pie图中label对应的列名
            * 仅当kind='pie'时有效

        values : string
            Pie图中的取值列名
            * 仅当kind='pie'时有效

        text : string
            包含text取值的列名

        keys : list of columns
            参与绘图的df列名，默认为df.columns
            也可以用作排序，作用类似于df[keys]

        ------------------------------------

        title : string
            图像标题

        xtitle : string
            x轴标题

        ytitle : string
            y轴标题

        y2title : string
            第二y轴名称

        ztitle : string
            z轴标题
            仅在3d图中生效

        ------------------------------------

        theme : string
            Layout 主题
                bowaer
                solar
                pearl
                white
                …………
            通过blot.getThemes() 查看所有内置可用主题，默认主题为bowaer

        colors : dict, list or string
            {key:color} 为每一条trace单独指定颜色
            [colors] zip(colors, traces)来指定颜色

        colorscale : string
            内置的色阶名称
            若色阶名前带'-'，则颜色反转
            仅当未设置colors参数时有效
            在blot.colors.scales()查看有效色阶名称

        fill : bool, str, list or dict
                bool: False不填充。True：'tozeroy' if kind=='area' else 'tonexty'
                str: 填充方式，取值列表：( "none" | "tozeroy" | "tozerox" | "tonexty" | "tonextx" | "toself" | "tonext" )
                list：单独为每条曲线指定填充方式
                dict：同上，key of keys
            填充曲线的方式，具体参数含义参照https://plotly.com/python/reference/scatter/#scatter-fill

        fillcolor : str, list or dict
                str: 填充颜色，和color方式一样。hex(e.g. '#ff0000')、rgb/rgba(e.g. 'rgba(255,0,0,0.8)')、css color name……
                list：单独为每条曲线指定填充颜色
                dict：同上，key of keys
            曲线填充的颜色控制

        width : dict, list or int
                int : 应用于所有曲线
                list : 按照顺序应用在各条曲线
                dict: {column:value} 为每条曲线单独指定线宽
            线宽，如果不指定，则从theme读取，如果theme没有，则默认为2

        dash : dict, list or string
                string : 应用于所有曲线
                list : 按照顺序应用在各条曲线
                dict: {column:value} 为每条曲线单独指定线宽
            线型
                solid「默认」：实线  '————'
                dash：短线 '- - - - '
                dashdot：带点短线 '-•-•-•-'
                dot：点连线  '••••••'

        mode : dict, list or string
                string : 应用于所有曲线
                list : 按照顺序应用在各条曲线
                dict: {column:value} 为每条曲线单独指定线宽
            散点图下绘制模式
                markers「默认」
                lines：连线。和kind='line'效果相同
                lines+markers：用线将散点连接
                lines+text：线+注解。text由text参数指定
                markers+text：点+注解
                lines+markers+text：线+点+注解

        interpolation : string
            连线方式
                linear「默认」：直线连接
                spline：曲线连接
                vhv：纵横纵的方式连接
                hvh：横纵横的方式连接
                vh：纵横的方式连接
                hv：横纵的方式连接

        symbol : string
            散点的形状，仅当mode的取值包含markers时生效
                circle「默认」：实心圆点
                circle-dot：空心圆点
                diamond：菱形
                square：正方形
                ………可参照plotly scatter symbol查看更多有效值

        size : string or int
            散点大小
                string：气泡图中气泡大小对应的列名
                int：散点图中散点的大小，定值

        ------------------------------------

        barmode : string
            多维柱状图展示形式，默认group
                group
                stack
                overlay
            仅当kind='bar'时有效

        bargap : float[0,1)
            组间间隔
            仅当kind='bar'或者'histogram'时有效

        bargroupgap : float[0,1)
            组内间隔距离
            仅当kind='bar'或者'histogram'时有效

        bins : int or tuple
            int: 分桶数量
            tuple: (start, end, size)
                start : 横坐标起始截断值
                end: 横坐标结束截断值
                size: 分桶数量
            仅在kind=histogram时生效

        histnorm : string
                '' (frequency)「默认」：频数
                percent: 百分数（0-100）
                probability：频率（0-1）
                density: 密度 = frequency/bins
                probability density: 概率密度 = probability/bins
            仅在kind=histogram时生效

        boxpoints : string
            箱形图（box）中散点的展示方式。默认不展示
                False「默认」：不展示
                outliers：仅展示离群点
                all：展示所有点
                suspectedoutliers：仅展示疑似离群点

        ------------------------------------

        orientation : string
            图例方向
                v「默认」：纵向
                h：横向
            仅当kind='bar'、'histogram'、'box'时有效

        annotations : dict
            图例标注
            {x_point : text}：{横坐标取值, 标注内容}

        vlines: list
            需要绘制垂直（平行于y轴）参考线的x取值列表

        hlines: list
            需要绘制水平（平行于x轴）参考线的y取值列表

        ------------------------------------

        gridcolor : string
            网格线颜色

        zerolinecolor : string
            坐标轴颜色

        margin : dict or tuple
                Tuple: 按照（left, right, bottom, top) 给出取值
                dict: key of ('l', 'r', 'b', 't'), 例如{'l':30} 代表距离左边界30px
            图例和上下左右边界的距离

        dimensions : tuple(int,int)
            图像的尺寸(width,height)
            jupyter中默认会充满输出
            返回静态图或者plotly对象默认为(800, 500)

        subplots : bool or tuple(rows, cols)
                bool:是否以子图的形式展示，默认为False。True则按照2列的形式展示子图
                tuple(rows, cols): 子图的栅栏分布
            子图的展示控制

        as_figure : bool=True
            是否返回plotly对象


    ==========Other Kwargs===================
        Line, Scatter
            connectgaps : bool=False
               折线图中空值是否连接

        ------------------------------------

        Pie charts
            sort : bool=True
                legend中各label是否按照占比排序

            pull : float [0-1] or list(float)
                    float：每块离中心的距离，默认为0
                    list：按顺序指定每个块离中心的距离
                饼图中块距离中心点的距离

            hole : float [0-1]
                饼图中心孔的尺寸，默认为0

            textcolor : string
                饼图中显示的文字颜色，默认自适应（深色背景浅色字体，浅色背景深色背景）

            textposition : string or list（string）
                饼图中的文字显示的位置 ['inside', 'outside', 'auto', 'none']

            textinfo : string
                饼图中文字显示的信息，默认为percent
                可以为['label', 'text', 'value', 'percent']中的一个
                或者多个用'+'连接的字符串，eg. 'label+value'

        ------------------------------------

        Heatmap and Surface
            center_value : float
                指定热力图scale的中心取值
                如果不指定，将自动从(zmin,zmax)中计算

            zmin : float
                指定计算colorscale最小的取值。eg. zmin=0.5, 则z<=0.5的颜色均为colorscale的起始颜色
                会影响到整个colorscale range

            zmax : float
                 指定计算colorscale最大的取值
                 会影响到整个colorscale range

        ------------------------------------

        Subplots
            horizontal_spacing : float [0,1]
                多个水平子图之间的距离，列间距

            vertical_spacing : float [0,1]
                多个竖直子图之间的距离，行间距

            subplot_titles : bool=False
                是否显示每个子图的title（默认使用legend信息）

            shared_xaxes、shared_xaxis : bool
                子图间是否共享x轴

            shared_yaxes : bool
                子图间是否共享y轴

        ------------------------------------

        Shapes
            hpsans : tuple(y0,y1) or list(tuple)
                填充水平（平行于y轴）区域，y0,y1表示区域的起始和结束纵坐标。可以用list传入多个

            vspans : tuple(x0,x1) or list(tuple)
                填充竖直（平行于x轴）区域，y0,y1表示区域的起始和结束纵坐标。可以用list传入多个

            shapes : dict or list(dict)
                plotly中的shape属性字典.具体参照https://plotly.com/python/shapes/#vertical-and-horizontal-lines-positioned-relative-to-the-axis-data
                eg. {"type":"rect", 'x0':1, 'y0':1, 'x1':2, 'y1':3}则绘制一个长方形框

        ------------------------------------

        Axis Ranges
            xrange : [lower_bound,upper_bound]
                指定x轴取值范围

            yrange : [lower_bound,upper_bound]
                指定y轴取值范围

            zrange : [lower_bound,upper_bound]
                指定z轴取值范围

        ------------------------------------

        Range Slider
            rangeslider : bool or dict
                    bool: 默认False, 如果为True，将展示默认的range slider
                    dict: 具体参照plotly文档：https://plotly.com/python/range-slider/
                        eg.  {'bgcolor':('blue',.3),'autorange':True}
                展示range slider

        ------------------------------------

        Annotations
            fontcolor : str
                注解文字颜色

            fontsize : int
                注解文字大小

            textangle : int=0
                注解文字方向
            完成注解相关配置参考：https://plot.ly/python/reference/#layout-annotations

    """
    # Valid Kwargs
    valid_kwargs = [
        'color', 'opacity', 'column', 'columns', 'labels', 'text', 'world_readable', 'colorbar',
        'vline_color', 'vline_width', 'hline_color', 'hline_width'
    ]
    BUBBLE_KWARGS = ['abs_size']
    TRACE_KWARGS = ['hoverinfo', 'connectgaps']
    HEATMAP_SURFACE_KWARGS = ['center_value', 'zmin', 'zmax']
    PIE_KWARGS = ['sort', 'pull', 'hole', 'textposition', 'textinfo', 'linecolor', 'linewidth', 'textcolor']
    OHLC_KWARGS = [
        'up_color', 'down_color', 'open', 'high', 'low', 'close',
        'volume', 'name', 'decreasing', 'increasing'
    ]
    SUBPLOT_KWARGS = [
        'horizontal_spacing', 'vertical_spacing',
        'specs', 'insets', 'start_cell', 'shared_xaxes',
        'shared_yaxes', 'subplot_titles', 'shared_xaxis', 'shared_yaxis'
    ]

    EXPORT_KWARGS = ['display_image', 'scale']

    kwargs_list = [
        tools.__LAYOUT_KWARGS, BUBBLE_KWARGS, TRACE_KWARGS,
        OHLC_KWARGS, PIE_KWARGS, HEATMAP_SURFACE_KWARGS, SUBPLOT_KWARGS,
        EXPORT_KWARGS,
    ]
    [valid_kwargs.extend(_) for _ in kwargs_list]

    dict_modifiers_keys = ['line']
    dict_modifiers = {}

    for k in dict_modifiers_keys:
        dict_modifiers[k] = kwargs_from_keyword(kwargs, {}, k, False)

    for key in list(kwargs.keys()):
        if key not in valid_kwargs:
            raise Exception("Invalid keyword : '{0}'".format(key))

    # Setting default values
    if not colors:
        colors = kwargs['color'] if 'color' in kwargs else colors
    if isinstance(colors, str):
        colors = [colors]
    opacity = kwargs.get('opacity', 0.8)

    theme = theme or 'bowaer'

    theme_config = tools.getTheme(theme)

    colorscale = colorscale or theme_config.get('colorscale', 'dflt')

    if width is None and kind != 'pie':
        width = theme_config.get('linewidth', 2)

    # 常见错误参数修正
    if 'column' in kwargs:
        keys = [kwargs['column']] if isinstance(kwargs['column'], str) else kwargs['column']
    if 'columns' in kwargs:
        keys = [kwargs['columns']] if isinstance(kwargs['columns'], str) else kwargs['columns']
    kind = 'line' if kind == 'lines' else kind

    # Figure generators
    def get_marker(marker):
        marker = marker or {}
        if 'line' in dict_modifiers:
            if 'color' not in dict_modifiers['line']:
                if 'linecolor' in kwargs:
                    linecolor = kwargs.get('linecolor')
                else:
                    if 'linecolor' in tools.getTheme(theme=theme):
                        linecolor = normalize(tools.getTheme(theme=theme)['linecolor'])
                    else:
                        linecolor = tools.getLayout(theme=theme)['xaxis']['titlefont']['color']
                dict_modifiers['line']['color'] = linecolor
            dict_modifiers['line'] = tools.updateColors(dict_modifiers['line'])
            marker['line'] = deep_update(marker['line'], dict_modifiers['line'])
        return marker

    # We assume we are good citizens
    validate = True

    if not layout:
        l_kwargs = dict([(k, kwargs[k]) for k in tools.__LAYOUT_KWARGS if k in kwargs])
        if annotations:
            ann_kwargs = check_kwargs(kwargs, tools.__ANN_KWARGS, clean_origin=False)
            annotations = tools.get_annotations(self.copy(), annotations, kind=kind, theme=theme, **ann_kwargs)

        layout = tools.getLayout(
            kind=kind, theme=theme, xTitle=xtitle, yTitle=ytitle, zTitle=ztitle, title=title,
            barmode=barmode, bargap=bargap, bargroupgap=bargroupgap, annotations=annotations,
            gridcolor=gridcolor, dimensions=dimensions, zerolinecolor=zerolinecolor,
            margin=margin, is3d='3d' in kind, **l_kwargs
        )
    elif isinstance(layout, Layout):
        layout = layout.to_plotly_json()

    layout['shapes'] = layout.get('shapes', [])
    layout['shapes'] += tools.add_ref_line(
        vlines or [], direction='vertical',
        color=kwargs.get('vline_color', 'red'), width=kwargs.get('vline_width', 2)
    )

    layout['shapes'] += tools.add_ref_line(
        hlines or [], direction='horizontal',
        color=kwargs.get('hline_color', 'red'), width=kwargs.get('hline_width', 2)
    )

    if not data:
        if kind in ('scatter', 'spread', 'ratio', 'bar', 'barh', 'area', 'line'):
            df = self.copy()
            if type(df) == pd.core.series.Series:
                df = pd.DataFrame({df.name: df})
            if x:
                df = df.set_index(x)
            if y and y2:
                _y = [y] if not isinstance(y, list) else y
                _secondary_y = [y2] if not isinstance(y2, list) else y2
                df = df[_y + _secondary_y]
            elif y:
                df = df[y]
            if kind == 'area':
                df = df.transpose().fillna(0).cumsum().transpose()
            mode = mode or ('lines' if kind != 'scatter' else 'markers')
            if text:
                if not isinstance(text, list):
                    text = self[text].values
            data = df.to_iplot(colors=colors, colorscale=colorscale, kind=kind, interpolation=interpolation,
                               fill=fill, fillcolor=fillcolor, width=width, dash=dash, keys=keys,
                               asDates=False, mode=mode, symbol=symbol, size=size,
                               text=text, **kwargs)
            trace_kw = check_kwargs(kwargs, TRACE_KWARGS)
            for trace in data:
                trace.update(**trace_kw)

            if kind in ('spread', 'ratio'):
                if kind == 'spread':
                    trace = self.apply(lambda x: x[0] - x[1], axis=1)
                    positive = trace.apply(lambda x: x if x >= 0 else np.nan)
                    negative = trace.apply(lambda x: x if x < 0 else np.nan)
                    trace = pd.DataFrame({'positive': positive, 'negative': negative})
                    trace = trace.to_iplot(colors={'positive': 'green', 'negative': 'red'}, width=0.5)
                else:
                    trace = self.apply(lambda x: x[0] * 1.0 / x[1], axis=1).to_iplot(colors=['green'], width=1)
                for t in trace:
                    t.update({'xaxis': 'x2', 'yaxis': 'y2', 'fill': 'tozeroy',
                              'name': kind.capitalize(), 'connectgaps': False, 'showlegend': False})
                data.append(trace[0])
                if kind == 'spread':
                    data.append(trace[1])
                layout['yaxis'].update({'domain': [.3, 1]})
                layout['yaxis2'] = copy.deepcopy(layout['yaxis'])
                layout['xaxis2'] = copy.deepcopy(layout['xaxis'])
                layout['yaxis2'].update(domain=[0, .25], title=kind.capitalize())
                layout['xaxis2'].update(anchor='y2', showticklabels=False)
                layout['hovermode'] = 'x'
            if 'bar' in kind:
                if 'stack' in barmode:
                    layout['legend'].update(traceorder='normal')
                orientation = 'h' if kind == 'barh' else orientation
                for trace in data:
                    trace.update(orientation=orientation)
                    if orientation == 'h':
                        trace['x'], trace['y'] = trace['y'], trace['x']

        elif kind == 'bubble':
            mode = 'markers' if 'markers' not in mode else mode
            x = self[x].values.tolist()
            y = self[y].values.tolist()
            z = size if size else z
            rg = self[z].values
            if not kwargs.get('abs_size', False):
                if len(rg) > 1:
                    z = [int(100 * (float(_) - rg.min()) / (rg.max() - rg.min())) + 12 for _ in rg]
                else:
                    z = [12] if len(rg) else []
            else:
                z = rg
            text = kwargs['labels'] if 'labels' in kwargs else text
            labels = self[text].values.tolist() if text else ''
            clrs = colors if colors else get_scales(colorscale)
            clrs = [clrs] if not isinstance(clrs, list) else clrs
            clrs = [clrs[0]] * len(x) if len(clrs) == 1 else clrs
            marker = dict(color=clrs, size=z, symbol=symbol,
                          line=dict(width=width))
            trace = Scatter(x=x, y=y, marker=marker, mode='markers', text=labels)
            data = [trace]

        elif kind in ('box', 'histogram', 'hist'):
            if isinstance(self, pd.core.series.Series):
                df = pd.DataFrame({self.name: self})
            else:
                df = self.copy()
            data = []
            clrs = get_colors(colors, colorscale, df.columns)
            if 'hist' in kind:
                barmode = 'overlay' if barmode == '' else barmode
                layout.update(barmode=barmode)
            columns = keys if keys else df.columns
            for _ in columns:
                if kind == 'box':
                    __ = Box(y=df[_].values.tolist(), marker=dict(color=clrs[_]), name=_,
                             line=dict(width=width), boxpoints=boxpoints)
                    # 114 - Horizontal Box
                    __['orientation'] = orientation
                    if orientation == 'h':
                        __['x'], __['y'] = __['y'], __['x']

                else:
                    __ = dict(x=df[_].values.tolist(), name=_,
                              marker=dict(color=clrs[_], line=dict(width=width)),
                              orientation=orientation,
                              opacity=kwargs['opacity'] if 'opacity' in kwargs else .8,
                              histnorm=histnorm)

                    __['marker'] = get_marker(__['marker'])

                    if orientation == 'h':
                        __['y'] = __['x']
                        del __['x']
                    __ = Histogram(__)
                    if bins:
                        if type(bins) in (tuple, list):
                            try:
                                _bins = {'start': bins[0], 'end': bins[1], 'size': bins[2]}
                                if orientation == 'h':
                                    __.update(ybins=_bins, autobiny=False)
                                else:
                                    __.update(xbins=_bins, autobinx=False)
                            except:
                                print("Invalid format for bins generation")
                        else:
                            if orientation == 'h':
                                __.update(nbinsy=bins)
                            else:
                                __.update(nbinsx=bins)
                data.append(__)

        elif kind in ('heatmap', 'surface'):
            if x:
                x = self[x].values.tolist()
            else:
                if self.index.__class__.__name__ in ('PeriodIndex', 'DatetimeIndex'):
                    x = self.index.format()
                else:
                    x = self.index.values.tolist()
            y = self[y].values.tolist() if y else self.columns.values.tolist()
            z = self[z].values.tolist() if z else self.values.transpose()
            scale = get_scales('rdbu') if not colorscale else get_scales(colorscale)
            scale = [normalize(_) for _ in scale]
            colorscale = [[float(_) / (len(scale) - 1), scale[_]] for _ in range(len(scale))]
            center_value = kwargs.get('center_value', None)

            if is_list(z):
                zmin = min(z)
                zmax = max(z)
            else:
                zmin = z.min()
                zmax = z.max()

            if center_value is not None:
                if center_value <= zmin + (zmax - zmin) / 2:
                    zmin = center_value * 2 - zmax
                else:
                    zmax = center_value * 2 - zmin
            zmin = kwargs.get('zmin', zmin)
            zmax = kwargs.get('zmax', zmax)
            if kind == 'heatmap':
                data = [Heatmap(z=z, x=x, y=y, zmin=zmin, zmax=zmax, colorscale=colorscale)]
            else:
                data = [Surface(z=z, x=x, y=y, colorscale=colorscale)]

        elif kind in ('scatter3d', 'bubble3d'):
            data = []
            keys = self[text].values if text else list(range(len(self)))
            colors = get_colors(colors, colorscale, keys, asList=True)
            mode = 'markers' if 'markers' not in mode else mode
            df = self.copy()
            df['index'] = keys
            if kind == 'bubble3d':
                rg = self[size].values
                if not kwargs.get('abs_size', False):
                    size = [int(100 * (float(_) - rg.min()) / (rg.max() - rg.min())) + 12 for _ in rg]
                else:
                    size = rg
            else:
                size = [size for _ in range(len(keys))]

            _data = Scatter3d(x=df[x].values.tolist(), y=df[y].values.tolist(), z=df[z].values.tolist(), mode=mode,
                              text=keys,
                              marker=dict(color=colors, symbol=symbol, size=size, opacity=.8))
            if text:
                _data.update(text=keys)
            data.append(_data)

        elif kind == 'pie':
            if not labels:
                raise BlotError('Missing: labels')
            if not values:
                raise BlotError('Missing: values')
            labels = self[labels].values.tolist()
            values = self[values].values.tolist()
            marker = dict(colors=get_colors(colors, colorscale, labels, asList=True))
            marker.update(line=dict(color=kwargs.pop('linecolor', None), width=kwargs.pop('linewidth', width)))
            pie = dict(values=values, labels=labels, name='', marker=marker)

            kw = check_kwargs(kwargs, PIE_KWARGS)
            kw['textfont'] = {'color': kw.pop('textcolor', None)}
            pie.update(kw)
            data = []
            del layout['xaxis']
            del layout['yaxis']
            data.append(Pie(pie))
            validate = False

    filename = title or 'blot-chart'

    figure = {
        'data': data, 'layout': layout
    }

    # Check secondary axis
    if y2:
        figure = tools._set_axis(figure, y2, side='right')
        if y2title:
            figure.layout.yaxis2.title = y2title

    # Subplots
    if subplots:
        fig = tools.strip_figures(figure)
        kw = check_kwargs(kwargs, SUBPLOT_KWARGS)
        for _ in ['x', 'y']:
            if 'shared_{0}axes'.format(_) not in kw:
                kw['shared_{0}axes'.format(_)] = kw.pop('shared_{0}axis'.format(_), False)
        if 'subplot_titles' in kwargs:
            if kwargs['subplot_titles'] == True:
                kw['subplot_titles'] = [d['name'] for d in data]
            else:
                kw['subplot_titles'] = kwargs['subplot_titles']
        figure = tools.subplots(fig, shape, base_layout=layout, theme=theme, **kw)

    # Exports
    validate = False if 'shapes' in layout else validate

    if as_figure:
        return Figure(figure)
    else:
        return iplot(
            figure, validate=validate, filename=filename,
            dimensions=dimensions
        )


def get_colors(colors, colorscale, keys, asList=False):
    if type(colors) != dict:
        if not colors:
            if colorscale:
                colors = get_scales(colorscale, len(keys))
        clrgen = colorgen(colors, len(keys))
        if asList:
            colors = [next(clrgen) for _ in keys]
        else:
            colors = {}
            for key in keys:
                colors[key] = next(clrgen)
    return colors


def get_items_as_list(items, keys, items_names='styles'):
    """
    Returns a dict with an item per key

    Parameters:
    -----------
        items : string, list or dict
            Items (ie line styles)
        keys: list
            List of keys
        items_names : string
            Name of items
    """
    if type(items) != dict:
        if type(items) == list:
            if len(items) != len(keys):
                raise Exception('List of {0} is not the same length as keys'.format(items_names))
            else:
                items = dict(zip(keys, items))
        else:
            items = dict(zip(keys, [items] * len(keys)))
    return items


def _scatter_matrix(self, theme=None, bins=10, color='grey', size=2, asFigure=False, **iplot_kwargs):
    """
    Displays a matrix with scatter plot for each pair of
    Series in the DataFrame.
    The diagonal shows a histogram for each of the Series

    Parameters:
    -----------
        df : DataFrame
            Pandas DataFrame
        theme : string
            Theme to be used (if not the default)
        bins : int
            Number of bins to use for histogram
        color : string
            Color to be used for each scatter plot
        size : int
            Size for each marker on the scatter plot
        iplot_kwargs : key-value pairs
            Keyword arguments to pass through to `iplot`
    """
    sm = tools.scatter_matrix(self, theme=theme, bins=bins, color=color, size=size)
    if asFigure:
        return sm
    else:
        return iplot(sm, **iplot_kwargs)


def _figure(self, **kwargs):
    """
    Generates a Plotly figure for the given DataFrame

    Parameters:
    -----------
            All properties avaialbe can be seen with
            help(blot.pd.DataFrame.iplot)
    """
    kwargs['asFigure'] = True
    return self.iplot(**kwargs)


def _layout(self, **kwargs):
    """
    Generates a Plotly layout for the given DataFrame

    Parameters:
    -----------
            All properties avaialbe can be seen with
            help(blot.pd.DataFrame.iplot)
    """
    kwargs['asFigure'] = True
    return self.iplot(**kwargs)['layout']


# ONLINE
# py.iplot(filename,fileopt,sharing,world_readable)
# py.plot(filename,fileopt,auto_open,sharing,world_readable)
# py.image.ishow(figure,format,width,height,scale)
# py.image.save_as(figure,filename,format,width,height,scale)

# OFFLINE
# py_offline.iplot(figure,show_link,link_text,validate,image,filename,image_width,image_height,config)
# py_offline.plot(figure,show_link,link_text,validate,output_type,include_plotlyjs,filename,auto_open,image,image_filename,image_width,image_height,config)


def iplot(figure, validate=True, filename='', dimensions=None, **kwargs):
    """
    Plots a figure in IPython, creates an HTML or generates an Image

    figure : figure
        Plotly figure to be charted
    validate : bool
        If True then all values are validated before
        it is charted
    sharing : string
        Sets the sharing level permission
            public - anyone can see this chart
            private - only you can see this chart
            secret - only people with the link can see the chart
    filename : string
        Name to be used to save the file in the server, or as an image
    online : bool
        If True then the chart/image is rendered on the server
        even when running in offline mode.
    asImage : bool
            If True it returns an Image (png)
            In ONLINE mode:
                Image file is saved in the working directory
                    Accepts:
                        filename
                        dimensions
                        scale
                        display_image
            In OFFLINE mode:
                Image file is downloaded (downloads folder) and a
                regular plotly chart is displayed in Jupyter
                    Accepts:
                        filename
                        dimensions
    asUrl : bool
        If True the chart url/path is returned. No chart is displayed.
            If Online : the URL is returned
            If Offline : the local path is returned
    asPlot : bool
        If True the chart opens in browser
    dimensions : tuple(int,int)
        Dimensions for image
            (width,height)
    display_image : bool
        If true, then the image is displayed after it has been saved
        Requires Jupyter Notebook
        Only valid when asImage=True

    Other Kwargs
    ============
        legend : bool
            If False then the legend will not be shown
        scale : integer
            Increase the resolution of the image by `scale` amount
            Only valid when asImage=True
    """
    valid_kwargs = ['world_readable', 'legend', 'scale']

    for key in list(kwargs.keys()):
        if key not in valid_kwargs:
            raise Exception(f'Invalid keyword : {key}')

    if 'legend' in kwargs and 'layout' in figure:
        figure['layout'].update(showlegend=kwargs['legend'])

    # Filename Handling
    filename = filename or 'blot-chart'

    # Dimensions
    dimensions = dimensions or (800, 500)

    # Remove validation if shapes are present
    if 'layout' in figure:
        validate = False if 'shapes' in figure['layout'] else validate

    # iplot
    return offline.py_offline.iplot(
        figure, validate=validate, filename=filename
    )


def _ta_figure(self, **kwargs):
    """
    Generates a Plotly figure for the given DataFrame

    Parameters:
    -----------
            All properties avaialbe can be seen with
            help(blot.pd.DataFrame.iplot)
    """
    kwargs['asFigure'] = True
    return self.ta_plot(**kwargs)


def _ta_plot(self, study, periods=14, column=None, include=True, str='{name}({period})', detail=False,
             theme=None, sharing=None, filename='', asFigure=False, **iplot_kwargs):
    """
    Generates a Technical Study Chart

    Parameters:
    -----------
            study : string
                Technical Study to be charted
                    sma - 'Simple Moving Average'
                    rsi - 'R Strength Indicator'
            periods : int
                Number of periods
            column : string
                Name of the column on which the
                study will be done
            include : bool
                Indicates if the input column(s)
                should be included in the chart
            str : string
                Label factory for studies
                The following wildcards can be used:
                    {name} : Name of the column
                    {study} : Name of the study
                    {period} : Period used
                Examples:
                    'study: {study} - period: {period}'
            detail : bool
                If True the supporting data/calculations
                are included in the chart
            study_colors : string or [string]
                Colors to be used for the studies

        Study Specific Parameters
        -------------------------
        RSI
            rsi_upper : int (0,100]
                Level for the upper rsi band
                default : 70
            rsi_lower : int (0,100]
                Level for the lower rsi band
                default : 30
        CCI
            cci_upper : int
                Level for the upper cci band
                default : 100
            cci_lower : int
                Level for the lower cci band
                default : -100
        BOLL
            boll_std : int or float
                Number of standard deviations
        MACD
            fast_period : int
                Number of periods for the fast moving average
            slow_period : int
                Number of periods for the slow moving average
            signal_period : int
                Number of periods for the signal
        CORREL
            how : string
                Method for the correlation calculation
                    values
                    pct_cht
                    diff

    """

    if 'columns' in iplot_kwargs:
        column = iplot_kwargs.pop('columns')

    if 'period' in iplot_kwargs:
        periods = iplot_kwargs.pop('period')

    if 'world_readable' in iplot_kwargs:
        sharing = iplot_kwargs.pop('world_readable')

    if 'study_color' in iplot_kwargs:
        iplot_kwargs['study_colors'] = iplot_kwargs.pop('study_color')

    if sharing is None:
        sharing = auth.get_config_file()['sharing']
    if isinstance(sharing, bool):
        if sharing:
            sharing = 'public'
        else:
            sharing = 'private'
    iplot_kwargs['sharing'] = sharing
    if theme is None:
        theme = iplot_kwargs.pop('study_theme', auth.get_config_file()['theme'])

    if not filename:
        if 'title' in iplot_kwargs:
            filename = iplot_kwargs['title']
        else:
            filename = 'Plotly Playground {0}'.format(time.strftime("%Y-%m-%d %H:%M:%S"))

    def get_subplots(figures):
        shape = (len(figures), 1)
        layout = tools.get_base_layout(figures)
        subplots = tools.subplots(figures, shape=shape, shared_xaxes=True, base_layout=layout)
        if len(figures) == 2:
            subplots['layout']['yaxis']['domain'] = [.27, 1.0]
            subplots['layout']['yaxis2']['domain'] = [0, .25]
        return subplots

    def get_study(df, func, iplot_kwargs, iplot_study_kwargs, str=None, include=False, column=None, inset=False):
        df = df.copy()
        if inset:
            if not column:
                if isinstance(df, pd.DataFrame):
                    column = df.keys().tolist()
                else:
                    df = pd.DataFrame(df)
                    column = df.keys().tolist()
        if 'legend' in iplot_kwargs:
            iplot_study_kwargs['legend'] = iplot_kwargs['legend']
        fig_0 = df.figure(**iplot_kwargs)
        df_ta = func(df, column=column, include=False, str=str, **study_kwargs)
        kind = iplot_kwargs['kind'] if 'kind' in iplot_kwargs else ''
        iplot_study_kwargs['kind'] = 'scatter'
        iplot_study_kwargs['colors'] = iplot_study_kwargs.get('colors',
                                                              ['blue', 'green', 'red'] if study == 'dmi' else 'blue')
        fig_1 = df_ta.figure(theme=theme, **iplot_study_kwargs)
        if kind in ['candle', 'ohlc']:
            for i in fig_1['data']:
                i['x'] = [pd.Timestamp(_) for _ in i['x']]
        if inset:
            figure = tools.merge_figures([fig_0, fig_1]) if include else fig_1
        else:
            figure = get_subplots([fig_0, fig_1]) if include else fig_1
        return figure

    study_kwargs = {}
    iplot_study_kwargs = {}

    study_kwargs = check_kwargs(iplot_kwargs, __TA_KWARGS, {}, clean_origin=True)
    iplot_study_kwargs = kwargs_from_keyword(iplot_kwargs, {}, 'study')

    study_kwargs.update({'periods': periods})

    ta_func = eval('ta.{0}'.format(study))

    inset = study in ('sma', 'boll', 'ema', 'atr', 'ptps')
    figure = get_study(self, ta_func, iplot_kwargs, iplot_study_kwargs, include=include,
                       column=column, str=str, inset=inset)

    ## Add Bands
    if study in ('rsi', 'cci'):
        bands = {'rsi': (30, 70),
                 'cci': (-100, 100)}
        _upper = study_kwargs.get('{0}_upper'.format(study), bands[study][0])
        _lower = study_kwargs.get('{0}_lower'.format(study), bands[study][1])
        yref = 'y2' if include else 'y1'
        shapes = [tools.get_shape(y=i, yref=yref, color=j, dash='dash') for (i, j) in
                  [(_lower, 'green'), (_upper, 'red')]]
        figure['layout']['shapes'] = shapes

    # if study=='rsi':
    # 	rsi_upper=study_kwargs.get('rsi_upper',70)
    # 	rsi_lower=study_kwargs.get('rsi_lower',30)
    # 	yref='y2' if include else 'y1'
    # 	shapes=[tools.get_shape(y=i,yref=yref,color=j,dash='dash') for (i,j) in [(rsi_lower,'green'),(rsi_upper,'red')]]
    # 	figure['layout']['shapes']=shapes

    # if study=='cci':
    # 	cci_upper=study_kwargs.get('cci_upper',100)
    # 	cci_lower=study_kwargs.get('cci_lower',-100)
    # 	yref='y2' if include else 'y1'
    # 	shapes=[tools.get_shape(y=i,yref=yref,color=j,dash='dash') for (i,j) in [(cci_lower,'green'),(cci_upper,'red')]]
    # 	figure['layout']['shapes']=shapes

    ## Exports

    if asFigure:
        return figure
    else:
        return iplot(figure, sharing=sharing, filename=filename)


def _fig_iplot(self, validate=True, sharing=None, filename='',
               online=None, asImage=False, asUrl=False, asPlot=False,
               dimensions=None, display_image=True, **kwargs):
    """
    Plots a figure in IPython

    figure : figure
        Plotly figure to be charted
    validate : bool
        If True then all values are validated before
        it is charted
    sharing : string
        Sets the sharing level permission
            public - anyone can see this chart
            private - only you can see this chart
            secret - only people with the link can see the chart
    filename : string
        Name to be used to save the file in the server, or as an image
    online : bool
        If True then the chart is rendered on the server
        even when running in offline mode.
    asImage : bool
            If True it returns an Image (png)
            In ONLINE mode:
                Image file is saved in the working directory
                    Accepts:
                        filename
                        dimensions
                        scale
                        display_image
            In OFFLINE mode:
                Image file is downloaded (downloads folder) and a
                regular plotly chart is displayed in Jupyter
                    Accepts:
                        filename
                        dimensions
    asUrl : bool
        If True the chart url/path is returned. No chart is displayed.
            If Online : the URL is returned
            If Offline : the local path is returned
    asPlot : bool
        If True the chart opens in browser
    dimensions : tuple(int,int)
        Dimensions for image
            (width,height)
    display_image : bool
        If true, then the image is displayed after it has been saved
        Requires Jupyter Notebook
        onlh valide when asImage=True

    Other Kwargs
    ============

        legend : bool
            If False then the legend will not be shown
    """
    return iplot(self, validate=validate, sharing=sharing, filename=filename,
                 online=online, asImage=asImage, asUrl=asUrl, asPlot=asPlot,
                 dimensions=dimensions, display_image=display_image, **kwargs)


pd.DataFrame.to_iplot = _to_iplot
pd.DataFrame.scatter_matrix = _scatter_matrix
pd.DataFrame.figure = _figure
pd.DataFrame.layout = _layout
pd.DataFrame.ta_plot = _ta_plot
pd.DataFrame.blot = _iplot
pd.DataFrame.ta_figure = _ta_figure
pd.Series.ta_figure = _ta_figure
pd.Series.ta_plot = _ta_plot
pd.Series.figure = _figure
pd.Series.to_iplot = _to_iplot
pd.Series.blot = _iplot
Figure.iplot = _fig_iplot

Figure.save = Figure.write_image
