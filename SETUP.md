

```textmate
conda create --name udacity python=3.8 pytorch torchvision torchaudio -c pytorch
python -m ipykernel install --user --name=udacity
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

If that doesn't work (older Jupyter versions), you can also modify the notebook directly:

```textmate
sed -i '' 's/"kernelspec":\(.*\)"name": "[^"]*"/"kernelspec":\1"name": "udacity"/g' cnn_from_scratch.ipynb
```