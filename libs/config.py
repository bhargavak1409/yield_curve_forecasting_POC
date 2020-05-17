from collections import OrderedDict 
import mxnet as mx
from pathlib import Path

# parameters for deepAR model for different asset category
param_dict = {
    'treasury_yield': {
        'model_path': Path.cwd().joinpath("assets").joinpath("models").joinpath("treasury_yield_model"),
        'freq': "M",
        'prediction_length': 1,
        'context_length': 5,
        'num_layers': 2,
        'num_cells': 20,
        'epochs': 150,
        'learning_rate': 1e-2,
        'batch_size': 32,
        'num_batches_per_epoch': 10,
        'use_feat_dynamic_real': True,
        'ctx': mx.cpu(0)
    },
    'corp_bond_yield': {
        'model_path': Path.cwd().joinpath("assets").joinpath("models").joinpath("corp_bond_yield_model"),
        'freq': "M",
        'prediction_length': 1,
        'context_length': 5,
        'num_layers': 2,
        'num_cells': 20,
        'epochs': 150,
        'learning_rate': 1e-2,
        'batch_size': 32,
        'num_batches_per_epoch': 10,
        'use_feat_dynamic_real': True,
        'ctx': mx.cpu(0)
    },
    'swap_yield': {
        'model_path': Path.cwd().joinpath("assets").joinpath("models").joinpath("swap_yield_model"),
        'freq': "M",
        'prediction_length': 1,
        'context_length': 5,
        'num_layers': 2,
        'num_cells': 20,
        'epochs': 100,
        'learning_rate': 1e-2,
        'batch_size': 32,
        'num_batches_per_epoch': 5,
        'use_feat_dynamic_real': True,
        'ctx': mx.cpu(0)
    }
}

# dictionary to map assets to their maturity dates (for x-axis names of yield plots)
maturity_dict = OrderedDict({
    '3mo': '3 months',
    '6mo': '6 months',
    '1yr': '1 year',
    '2yr': '2 year',
    '3yr': '3 year',
    '4yr': '4 year',
    '5yr': '5 year',
    '7yr': '7 year',
    '10yr': '10 year'
})

# dictionary to map asset name to their csv file names
asset_name_dict = {
    'fed_funds':'Effective Fed Fund Rate',
    'HQM_10yr':'10 Year High Quality Market Corporate Bond Spot Rate',
    'HQM_1yr':'1 Year High Quality Market Corporate Bond Spot Rate',
    'HQM_3yr':'3 Year High Quality Market Corporate Bond Spot Rate',
    'HQM_5yr':'5 Year High Quality Market Corporate Bond Spot Rate',
    'HQM_6mo':'6 Month High Quality Market Corporate Bond Spot Rate',
    'ICE_10yr':'ICE Swap Rates, 10 Year Tenor',
    'ICE_1yr':'ICE Swap Rates, 1 Year Tenor',
    'ICE_2yr':'ICE Swap Rates, 2 Year Tenor',
    'ICE_3yr':'ICE Swap Rates, 3 Year Tenor',
    'ICE_4yr':'ICE Swap Rates, 4 Year Tenor',
    'ICE_5yr':'ICE Swap Rates, 5 Year Tenor',
    'treasury_10yr':'10 Year Treasury Constant Maturity Rate',
    'treasury_1yr':'1 Year Treasury Constant Maturity Rate',
    'treasury_3mo':'3 Month Treasury Constant Maturity Rate',
    'treasury_3yr':'3 Year Treasury Constant Maturity Rate',
    'treasury_5yr':'5 Year Treasury Constant Maturity Rate',
    'treasury_6mo':'6 Month Treasury Constant Maturity Rate'
}