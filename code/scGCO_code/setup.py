from setuptools import setup, find_packages
setup(
        name='scGCO',
        version='1.1.0',
        description='single-cell graph cuts optimization',
        url='https://github.com/WangPeng-Lab/scGCO',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['pandas','numpy','matplotlib','scipy','sklearn','seaborn','parmap','Cython','pygco',
                        'tqdm','networkx','shapely','statsmodels','hdbscan','pillow'],
        author='Peng Wang',
        author_email='wangpeng@picb.ac.cn',
        license='MIT'

)
