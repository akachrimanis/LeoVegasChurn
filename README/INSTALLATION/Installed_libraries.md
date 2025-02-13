````
source venv/bin/activate

# Basic
poetry add pandas matplotlib seaborn python-dotenv configparser joblib PyYAML
poetry add lifelines scikit-survival scikit-learn mlflow feast

# Jupyter lab
poetry add jupyterlab ipykernel jupyterlab-lsp
poetry add 'python-lsp-server[all]'

# Black background
poetry add --dev black
poetry run pip install "jupyterlab_code_formatter[lab]"

# create config file for jupyter
poetry run jupyter lab --generate-config    
# add code in config
c = get_config()
c.CodeFormatterManager.default_formatter = "black"


# code foramt
poetry run jupyter labextension install @ryantam626/jupyterlab_code_formatter
poetry run ipython kernel install --user --name=LeoVegas --display-name="LeoVegas 3.11 (LeoVegas)"

# Jupyter visualisations
poetry add sweetviz lux-api autoviz missingno qgrid
poetry run pip install kaleido      




poetry add nbstripout nbdime
nbdime config-git --enable

poetry add --group dev colorlog Werkzeug
````
# include at the end of the pyproject.toml
# Correct way to include the src folder
[tool.poetry]
packages = [
    { include = "your_library", from = "src" }
]

## Delete a kernel

````
jupyter kernelspec list                                             
````


# Fix the python path for the src file
````
source ~/.zshrc  # Or source ~/.bashrc
````