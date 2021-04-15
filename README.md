# Blot

> blot基于plotly和cufflinks，是一个进程安全，比plotly更简约，比cufflinks更快的python动态图可视化库。更贴近日常使用场景，便捷的绘制基带、简单的静态图像转换存储…结合pandas的Series和DataFrame使用（monkey patching），仅需一行代码即可完成各类图表构建。

## 安装方式
1. pip3 install blotly
2. 下载源码。进入项目根目录执行 python setup.py install

## 使用方式
1. 具体参照文档：Blot.pdf

## 版本历史
### v1.0.0
* 解决一些bug，支持多进程
* 精简项目，优化参数和注释
* 增加基带绘制和相关控制参数
* 支持便捷的静态图保存
* ……

## 示例
```python
import pandas as pd
import numpy as np
import blot

pd.DataFrame({
    'upper':np.random.randint(30,40,50),
    'lower':np.random.randint(10,20,50),
    'value1':np.random.randint(15,65,50),
    'value2':np.random.randint(0,20,50),
}).blot(upper='upper', lower='lower',color='blue')
```
![bounds](img/bounds.png)