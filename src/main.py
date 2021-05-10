from tabulate import tabulate
from src.helpers import get_filled_table

if __name__ == '__main__':

    result_table = get_filled_table()

    print(tabulate(result_table))
