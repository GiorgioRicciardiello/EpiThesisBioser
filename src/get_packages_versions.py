import importlib.metadata
import pandas as pd

# List of the packages you want to report
packages = [
    'pathlib',  # built-in, version N/A
    'os',       # built-in, version N/A
    'numpy',
    'pandas',
    'xgboost',
    'scikit-learn',
    'optuna',
    'matplotlib',
    'seaborn',
    'tqdm',
    'scipy',
    'fonttools',
    'statsmodels',
    'tabulate'
]

rows = []
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        version = 'N/A'
    rows.append({'package': pkg, 'version': version})

df_versions = pd.DataFrame(rows)

# Display as a Markdown table
print(df_versions.to_markdown(index=False))
