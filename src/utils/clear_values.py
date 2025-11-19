import pandas as pd

def clearvars():
    """
    Deletes all global variables from the caller's global namespace except built-in and dunder variables.
    Prints a message for each deleted variable.
    """
    for el in sorted(globals()):
        if '__' not in el:
            print(f'deleted: {el}')
            del el

# Automatically clear globals when this script runs as main module
if __name__ == "__main__":
    clearvars()

