def flatten_nested_list(nested_list):
    results = []
    for element in nested_list:
        if not isinstance(element, (list, tuple)):
            results.append(element)
        else:
            results.extend(flatten_nested_list(element))
    return results