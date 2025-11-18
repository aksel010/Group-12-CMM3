"""
clear_val.py
Helper utility to clear user-defined variables from global namespace.
"""

def clear_vars():
    """
    Delete all user-defined global variables (for interactive use).
    Prints each variable name as it is deleted.
    """
    for el in sorted(list(globals().keys())):
        if '__' not in el:
            print(f'deleted: {el}')
            del globals()[el]

if __name__ == '__main__':
    clear_vars()
