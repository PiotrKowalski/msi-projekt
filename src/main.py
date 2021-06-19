from tabulate import tabulate
from src.helpers import get_filled_table, transform_row_data_to_row_mean, transform_row_data_to_row_std, save_figs

if __name__ == '__main__':
    result_table = get_filled_table()

    print(tabulate(result_table))

    save_figs(result_table)
