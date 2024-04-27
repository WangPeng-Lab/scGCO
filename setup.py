from setuptools import setup, find_packages
setup(
        name='scGCO',
        version='1.1.3',
        description='single-cell graph cuts optimization',
        url='https://github.com/fengwanwan/scGCO',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['pandas','numpy','matplotlib','scipy','scikit-learn','seaborn','parmap','Cython','pygco',
                        'tqdm','networkx','shapely','statsmodels','hdbscan','pillow','scikit-image','umap-learn','pysal==2.0.0'],
        author='Wanwan Feng',
        author_email='fengwanwan2023@gmail.com',
        license='MIT'

)

## pip install numpy
## pip install pygco
## conda install -c conda-forge cython
## pip uninstall shapely && conda install -c r shapely   
