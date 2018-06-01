from config.base_config import cfg
from single_data_provider import SingleDataProvider


def get_data_provider(data_split, batchsize=1):
    if cfg.NTHREADS > 1:
        try:
            import torch
            from multi_data_provider import MultiDataProvider
        except:
            cfg.NTHREADS = 1


    if cfg.NTHREADS > 1:
        data_provider = MultiDataProvider(data_split, batchsize)
    else:
        data_provider = SingleDataProvider(data_split, batchsize)
    return data_provider