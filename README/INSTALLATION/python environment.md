````
poetry init
poetry config virtualenvs.in-project true
poetry env use /opt/homebrew/bin/python3.11  
poetry install --no-root
poetry env info --path

source /Users/tkax/dev/aimonetize/WIP/LeoVegasChurn/.venv/bin/activate


nano ~/.zshrc 
export PYTHONPATH="/Users/tkax/dev/aimonetize/WIP/LeoVegasChurn:$PYTHONPATH"
source ~/.zshrc

````

````
poetry cache clear pypi --all  # Clear the cache for PyPI packages
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

````