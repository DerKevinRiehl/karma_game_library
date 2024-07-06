# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from typing import Callable, List, Dict, Set
from numbers import Number

# Functions
def non_negative(value: Number, name: str):
    """
    This function checks if a given value is not negative.
    
    Parameters
    ----------
    value : Number
        The number to be checked.
    name : str
        The name of the number.
    
    Raises
    ----------
    Exception
        If the value is below zero.
    
    Returns
    -------
    None
    """
    if value < 0:
        raise Exception('"' + str(name) + '" must be greater than zero!')


def set_not_empty(set_instance: Set, name: str):
    """
    This function checks if a given set instance is empty or not.
    
    Parameters
    ----------
    set_instance : Set
        The set instance to be checked.
        
    name : str
        The name of the set instance.
        
    Raises
    ----------
    Exception
        If the set is empty.
        
    Returns
    -------
    None
    """
    if len(set_instance) == 0:
        raise Exception('"' + str(name) + '" must have at least one element!')


def map_keys(map_instance: Dict, lst_valid_keys: List, name: str, name2: str):
    """
    This function checks if a given map contains exactly the keys of a given
    list of keys.
    
    Parameters
    ----------
    map_instance : Dict
        The map instance to be checked.
    lst_valid_keys : List
        The list with valid keys.
    name : str
        The name of the map instance.
    name2 : str
        The name of the list instance.
    
    Raises
    ----------
    Exception
        If any key in the map does not appear in the list.
        If any key in the list does not appear in the map.
        
    Returns
    -------
    None
    """
    for key in map_instance:
        if key not in lst_valid_keys:
            raise Exception(
                'map "'
                + str(name)
                + '" key "'
                + str(key)
                + '" does not appear in "'
                + str(name2)
                + '"!'
            )
    for key in lst_valid_keys:
        if key not in map_instance:
            raise Exception(
                'key "'
                + str(key)
                + '" of set "'
                + str(name2)
                + '" does not appear in map "'
                + str(name)
                + '"!'
            )

def map_values_bounded(map_instance: Dict, min_val: float, max_val: float, name: str):
    """
    This function checks if the values of a given Map are in between a range
    of values.
    
    Parameters
    ----------
    map_instance : Dict
        The map instance to be checked.
    min_val : float
        The minimum allowed value.
    max_val : float
        The maximum allowed value.
    name : str
        The name of the map instance.
        
    Raises
    ----------
    Exception
        If any value of the map exceeds the defined range.
        
    Returns
    -------
    None
    """
    for key in map_instance:
        value = map_instance[key]
        if value < min_val:
            raise Exception(
                'map "'
                + str(name)
                + '" contains illegal value "'
                + str(value)
                + '" which is smaller than minimal allowed value "'
                + str(min_val)
                + '"!'
            )
        if value > max_val:
            raise Exception(
                'map "'
                + str(name)
                + '" contains illegal value "'
                + str(value)
                + '" which is larger than minimal allowed value "'
                + str(max_val)
                + '"!'
            )


def float_bounded(float_instance: float, min_val: float, max_val: float, name: str):
    """
    This function checks if a given float instance is in between a range of
    values.
    
    Parameters
    ----------
    float_instance : float
        The float value to be checked.
    min_val : float
        The minimum allowed value.
    max_val : float
        The maximum allowed value.
    name : str
        Name of the float instance.
        
    Raises
    ----------
    Exception
        If float_instance exceeds the defined range.
    
    Returns
    -------
    None
    """
    if float_instance < min_val:
        raise Exception(
            'float "'
            + str(name)
            + '" contains illegal value "'
            + str(float_instance)
            + '" which is smaller than minimal allowed value "'
            + str(min_val)
            + '"!'
        )
    if float_instance > max_val:
        raise Exception(
            'float "'
            + str(name)
            + '" contains illegal value "'
            + str(float_instance)
            + '" which is larger than minimal allowed value "'
            + str(max_val)
            + '"!'
        )


def list_length(lst_instance: List, length: int, name: str):
    """
    This function checks if a given list instance has a certain length 
    (number of elements).
        
    Parameters
    ----------
    lst_instance : List
        The list instance to be checked.
    length : int
        The lenght, number of elements.
    name : str
        Name of the list instance.
        
    Raises
    ----------
    Exception
        If lst_instance does not have specified length.
    
    Returns
    -------
    None
    """
    if len(lst_instance) != length:
        raise Exception(
            'list "'
            + str(name)
            + '" contains too few elements, population size was defined as "'
            + str(length)
            + '"!'
        )


def func_not_none(func: Callable, name: str):
    """
    This function checks if a given function is not None.
    
    Parameters
    ----------
    func : Callable
        A function to be checked.
        
    name : str
        Name of the function.
        
    Raises
    ----------
    NotImplementedError
        If function is None (not implemented).
    
    Returns
    -------
    None
    """
    if func is None:
        raise NotImplementedError('Function "' + str(name) + '" is not defined (None)!')
