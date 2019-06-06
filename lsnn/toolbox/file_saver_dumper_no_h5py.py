"""
Copyright (C) 2019 the LSNN team, TU Graz
"""

import json
import numpy as np
import os
import pickle
import datetime
from collections import OrderedDict
import tensorflow as tf

## Functions to process tensorflow flags
def flag_to_dict(FLAG):
    if float(tf.__version__[2:]) >= 5:
        flag_dict = FLAG.flag_values_dict()
    else:
        flag_dict = FLAG.__flags
    return flag_dict

def get_storage_path_reference(script_file, FLAG, root, flags=True, comment=True):

    # just evalute once the flag cause sometimes it is bugged
    key0 = list(dir(FLAG))[0]
    getattr(FLAG,key0)

    # SETUP THE SAVING FOLDER
    script_name = os.path.basename(script_file)[:-3]
    root_path = os.path.join(root,script_name)
    # File reference for saving info
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M__%S_%f")

    flag_dict = flag_to_dict(FLAG)
    assert isinstance(flag_dict,dict)

    random_key = str(np.random.randint(0,1000000)).zfill(6)
    file_reference = time_stamp + '-' + random_key
    if flags:
        config = OrderedDict(sorted((flag_dict.items())))
        string_list = [k + '_' + str(v) for k, v in config.items()]
        file_reference = file_reference + '-' + '-'.join(string_list)
    if comment:
        file_reference = file_reference + '__' + flag_dict["comment"]
    file_reference = file_reference[:240]
    full_storage_path = os.path.join(root_path,file_reference)
    return file_reference,full_storage_path, flag_dict

## JSON
class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyAwareEncoder, self).default(obj)



## GENERAL

def save_file(obj, path, file_name, file_type='pickle'):

    # Put the file type at the end if needed
    if not(file_name.endswith('.' + file_type)):
        file_name = file_name + '.' + file_type

    # Make sure path is provided otherwise do not save
    if path == '':
        print(('WARNING: Saving \'{0}\' cancelled, no path given.'.format(file_name)))
        return False

    if file_type == 'json':
        assert os.path.exists(path), 'Directory {} does not exist'.format(path)
        f = open(os.path.join(path, file_name), 'w')
        json.dump(obj, f, indent=4, sort_keys=True, cls=NumpyAwareEncoder)
        f.close()
    elif file_type == 'pickle':
        f = open(os.path.join(path, file_name), 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        raise NotImplementedError('SAVING FAILED: Unknown format {}'.format(pickle))

    return True

def load_file(path,file_name,file_type=None):

    if file_type is None:
        if file_name.endswith('.json'):
            file_type = 'json'
        elif file_name.endswith('.pickle'):
            file_type = 'pickle'
        else:
            raise ValueError('LOADING FAILED: is file type is None, file type should be given in file name. Got {}'.format(file_name))
    else:

        # Put the file type at the end if needed
        if not (file_name.endswith('.' + file_type)):
            file_name = file_name + '.' + file_type

    if path == '':
        print(('Saving \'{0}\' cancelled, no path given.'.format(file_name)))
        return False

    if file_type == 'json':
        f = open(os.path.join(path, file_name), 'r')
        obj = json.load(f)
    elif file_type == 'pickle':
        f = open(os.path.join(path, file_name), 'rb')
        obj = pickle.load(f)
    else:
        raise ValueError('LOADING FAILED: Not understanding file type: type requested {}, file name {}'.format(file_type,file_name))

    return obj

def compute_or_load(function,path,file_name,file_type='pickle',verbose=True):

    file_path = os.path.join(path, file_name + '.' + file_type)

    if os.path.exists(file_path):
        if verbose: print('File {} loaded'.format(file_name))
        return load_file(path,file_name,file_type= file_type)

    else:
        obj = function()
        save_file(obj, path, file_name, file_type=file_type)
        if verbose: print('File {} saved'.format(file_name))

    return obj

