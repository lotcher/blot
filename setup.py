from setuptools import setup
from os import path

BASE_DIR = path.dirname(path.abspath(__file__))
README_DIR = path.join(BASE_DIR, 'README.md')
with open(path.join(BASE_DIR, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().split()

setup(
    name='blotly',
    version='1.0.3',
    description='基于cufflinks的绘图工具',
    author='bowaer',
    author_email='cb229435444@outlook.com',
    license='MIT',
    keywords=['pandas', 'plotly', 'plotting'],
    url='https://github.com/lotcher/blot',
    packages=['blot'],
    package_data={'blot': ['../helper/*.json']},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    zip_safe=False,
    long_description=open(README_DIR, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    data_files=[README_DIR]
)
