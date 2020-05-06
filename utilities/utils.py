def remove_values_from_list(list: list, values: list):
    """
    Take a list and remove unnecessary values
    :param list: target list
    :param values: unnecesary values
    :return: list
    """
    for value in values:
        list.remove(value)
    return list
