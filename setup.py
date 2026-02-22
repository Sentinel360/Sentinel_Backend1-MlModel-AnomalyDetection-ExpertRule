from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sentinel360-backend",
    version="0.1.0",
    author="Caleb Okwesie Arthur, Frances Seyram Fiahagbe",
    author_email="caleb.arthur@ashesi.edu.gh",
    description="AI-powered passenger safety monitoring for ride-hailing in Ghana",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sentinel360/Sentinel_ML_Model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "googlemaps>=4.10.0",
        "shapely>=2.0.0",
        "pyproj>=3.6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
        ],
        "simulation": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
    },
)
