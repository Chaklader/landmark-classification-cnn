

```textmate
conda create --name udacity python=3.8 pytorch torchvision torchaudio -c pytorch
python -m ipykernel install --user --name=udacity

python -m ipykernel install --user --name udacity --display-name "Python (udacity)"
```

Yes, you can set the kernel for specific notebooks using the terminal with jupyter command. Here's how:

```textmate
# For a single notebook
jupyter kernelspec set-kernel --notebook=cnn_from_scratch.ipynb udacity

# For multiple notebooks
jupyter kernelspec set-kernel --notebook=cnn_from_scratch.ipynb --notebook=transfer_learning.ipynb udacity

# For all notebooks in current directory
jupyter kernelspec set-kernel --notebook="*.ipynb" udacity
```


The issue might be related to how your Python package is structured. Since your code uses relative imports (from .helpers 
import compute_mean_and_std), you need to make sure your package is installed properly.

Here's what you can try:

1. First, make sure you have a setup.py file in your root directory. If not, create one:
```python
from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
)
```

2. Then, in your root directory (where setup.py is), install your package in development mode:
```bash
pip install -e .
```

3. Your directory structure should look something like this:
```
landmark-classification-cnn/
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── helpers.py
└── notebooks/
    └── cnn_from_scratch.ipynb
```

4. Make sure you have an empty __init__.py file in your src directory.

5. After doing this, try restarting the kernel in your notebook and running the code again.

If you're still getting the ModuleNotFoundError, try importing the modules directly instead of using relative imports. Change:
```python
from .helpers import compute_mean_and_std, get_data_location
```
to:
```python
from src.helpers import compute_mean_and_std, get_data_location
```

Let me know if any of these steps helps resolve the issue.


If that doesn't work (older Jupyter versions), you can also modify the notebook directly:

```textmate
sed -i '' 's/"kernelspec":\(.*\)"name": "[^"]*"/"kernelspec":\1"name": "udacity"/g' cnn_from_scratch.ipynb
```