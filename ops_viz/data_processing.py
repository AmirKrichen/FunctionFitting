import pandas as pd
import numpy as np


class MathUtils:
    """
    A class for mathematical functions used in data processing.
    """
    def sqd_dev_sum(self, column1, column2):
        """
        Calculates the squared deviation sum for the specified columns

        :return: The sum of squared deviations between the two columns.
        """
        return sum((column1 - column2) ** 2)

    def max_deviation(self, column1, column2):
        """
        Calculates the maximum deviation for the specified columns

        :return: The maximum deviation between the two columns.

        """
        return max(abs(column1 - column2))


class DataHandler:
    """
    Base Class for handling data loading
    """
    def __init__(self, session):
        self.session = session

    def get_data(self, table):
        """
        Loads a database table into a DataFrame using SQLAlchemy's session.

        :return: A Pandas DataFrame with data from the specified table
        """
        with self.session as session:
            try:
                return pd.read_sql_table(table, session.bind)
            except Exception as e:
                print(f"Fetching {table} data failed. Error occurred: {e}")
            except pd.errors.DatabaseError as e:
                print(f"Error retrieving {table} data: {e}")

    def get_csv_data(self, file_path):
        """
        Loads a CSV file into a Pandas DataFrame.

        :return: A Pandas DataFrame with data from the specified file
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            print(f"CSV file not found: {e.filename}")
            exit()
