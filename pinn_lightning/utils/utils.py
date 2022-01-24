def flatten_nested_list(nested_list):
    results = []
    for element in nested_list:
        if not isinstance(element, (list, tuple)):
            results.append(element)
        else:
            results.extend(flatten_nested_list(element))
    return results


def convert_to_list(arg, length=1, assertion=False):
    if arg is None:
        arg = []

    elif not isinstance(arg, (list, tuple)):
        arg = [arg for _ in range(length)]

    else:
        if assertion:
            assert len(arg) == length
    
    return arg