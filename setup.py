from setuptools import setup
from os import path
import io

here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requires = f.read().split()

setup(
    name='blotly',
    version='1.0.0',
    description='基于cufflinks的绘图工具',
    author='bowaer',
    author_email='cb229435444@outlook.com',
    license='MIT',
    keywords=['pandas', 'plotly', 'plotting'],
    url='https://github.com/lotcher/blot',
    packages=['blot'],
    package_data={'blot': ['../helper/*.json']},
    include_package_data=True,
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    zip_safe=False
)
