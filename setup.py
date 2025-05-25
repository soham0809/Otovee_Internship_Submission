from setuptools import find_packages, setup

setup(
    name='pcos-ml-project',
    version='0.0.1',
    author='Soham Joshi',
    author_email='sohamj69@gmail.com',
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
