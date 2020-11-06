# -*-coding:utf-8 -*-
"""gklearn - datasets module

Implement some methods to manage graph datasets
 graph_fetcher.py : fetch graph datasets from the Internet.


"""

# info
__version__ = "0.2"
__author__ = "Linlin Jia"
__date__ = "October 2020"


from gklearn.dataset.metadata import DATABASES, DATASET_META
from gklearn.dataset.metadata import GREYC_META, IAM_META, TUDataset_META
from gklearn.dataset.metadata import list_of_databases, list_of_datasets
from gklearn.dataset.data_fetcher import DataFetcher