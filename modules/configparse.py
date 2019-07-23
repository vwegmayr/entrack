"""
Replace parts of yaml config with Python objects.
"""
import importlib
import importlib.util
import os
import yaml
import argparse

class ConfigParser:
    """Parser for yaml files"""

    def __init__(self):
        self.config = None

    def parse_raw(self, filename):

        filename = os.path.normpath(os.path.expanduser(filename))
        with open(filename) as config_file:
            raw_config = yaml.load(config_file)

        return raw_config

    def parse(self, filename):
        """Parse a yaml file.
        Args:
            filename (str): Path to the yaml file to be parsed.
        Returns:
            dict: Python dict with same contents as yaml file,
                but parts replaced with objects as specified
                in parse_python_objects.
        """
        config_dict = self.parse_raw(filename)
        # Replace parts of config_dict yaml in place
        self.parse_python_objects(config_dict)
        return config_dict

    def parse_python_objects(self, yaml_dict):
        """Recursively replace parts of yaml-style dict with python objects (in place).
        Args:
            yaml_dict: Yaml-style Python dict.
        """
        if isinstance(yaml_dict, dict):

            # Add custom replacements here
            replace_obj_from_module([
                "_fn",
                "_func",
                "class",
                "activation",
                "_op"], yaml_dict)

            for key, value in yaml_dict.items():
                self.parse_python_objects(value)

        elif isinstance(yaml_dict, list):
            for item in yaml_dict:
                self.parse_python_objects(item)


def import_obj_from_string(string):
    module_string = ".".join(string.split(".")[:-1])
    module = importlib.import_module(module_string)
    obj_key = string.split(".")[-1]
    return getattr(module, obj_key)


def replace_obj_from_module(strings, dict):
    """Replace string values with objects, in place.

    If a key in dict contains string, the value is replaced with
    the object described by the value, which is assumed to be in
    the form 'module.submodule.object'.

    Args:
        string (str): String to look for.
        dict (dict): Dictionary being parsed.
    """
    for string in strings:
        if any_key_contains(string, dict):
            full_keys = get_full_keys_containing(string, dict)
            for full_key in full_keys:
                if isinstance(dict[full_key], str):
                    dict[full_key] = import_obj_from_string(dict[full_key])


def any_key_contains(string, dict):
    for key in dict.keys():
        if string in key:
            return True
    return False


def get_full_keys_containing(string, dict):
    keys = []
    for key in dict.keys():
        if string in key:
            keys.append(key)
    return keys


def parse_more_args(more_args):
    """Parse unknown args returned by argparse

    Args:
        more_args (list): List which contains the args
            in the form ["--arg1=val1", "--arg2=val2", ...].

    Returns:
        parsed_args (argpars.Namespace)
    """
    
    if len(more_args) > 0:
        arg_dict = {}
        for arg in more_args:
            key, val = arg.split("=")
            assert key[:2] == "--"
            key = key[2:]
            try:
                arg_dict[key] = int(val)
            except ValueError:
                try:
                    arg_dict[key] = float(val)
                except ValueError:
                    if val == "True":
                        arg_dict[key] = True
                    elif val == "False":
                        arg_dict[key] = False
                    else:
                        arg_dict[key] = val

        return argparse.Namespace(**arg_dict)
    else:
        return argparse.Namespace(**{})
