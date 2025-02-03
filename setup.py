from setuptools import setup, find_packages

setup(
    name="bakery_sales_prediction",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
) 