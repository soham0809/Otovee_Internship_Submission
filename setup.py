from setuptools import find_packages, setup

setup(
    name='pcos-ml-project',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'flask',
        'joblib',
        'gunicorn'
    ]
)
