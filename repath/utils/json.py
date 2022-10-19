import pandas as pd


def to_json(obj):
    otype = type(obj)
    if otype == pd.DataFrame:
        
    return {'type': otype,
            'fields': an_object.__dict__}


def from_json(json_object):
    if 'type' in json_object:
        type_name = json_object['type']
        kargs_dict = json_object['fields']
        return eval(type_name)(**kargs_dict)
    return json_object