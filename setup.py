from setuptools import setup
import setuptools

setup(
    name='edu-card-analyser',
    version='1.2.0',
    author='Educatena',
    author_email='tic@educatena.com.br',
    description='Educatena\'s Computer Vision python package for analysing Answer Cards from images.',
    url='https://github.com/educatena/edu-card-analyser',
    # packages=['example_package'],
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6'
)
