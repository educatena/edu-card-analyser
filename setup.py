from setuptools import setup
import setuptools

from edu_card_models.SheetV1 import SheetV1

setup(
    name='edu-card-analyser',
    version='0.0.1',
    author='Educatena',
    author_email='...',
    description='Educatena\'s Computer Vision python package for analysing Answer Cards from images.',
    url='https://github.com/pypa/sampleproject',
    # packages=['example_package'],
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6'
)
