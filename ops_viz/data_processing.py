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


class ProcessData(DataHandler):
    """
    A class for processing and analyzing data.

    This class provides methods to
        - Assigns and ideal functions to each train Function (least square)
        - Maps individual test Data to one of the four selected ideal Functions
    """
    def __init__(self, test_path, session):
        super().__init__(session)
        self.test_path = test_path
        self.math = MathUtils()

    def select_functions(self):
        """
        Selects the 4 ideal functions which have the minimum sum of all
        y-deviations squared then calculates their maximum deviation.

        :return: A dictionary mapping training data columns to their selected
        ideal function.
        """
        # loads data from database
        ideal_data = self.get_data('ideal_functions')
        train_data = self.get_data('train_data')

        self.selection = {}
        # Loops through the train_data table columns Y1,..,Y4
        for train_column in train_data.columns[1:]:
            least_squared = float('inf')

            # Loops through the ideal_functions table columns Y1,..,Y50
            for ideal_column in ideal_data.columns[1:]:
                sqd_sum = self.math.sqd_dev_sum(train_data[train_column],
                                                ideal_data[ideal_column])
                if least_squared > sqd_sum:
                    # Finds the function with the least squared error
                    least_squared = sqd_sum
                    # Finds max deviation between train data and ideal function
                    max_dev = self.math.max_deviation(train_data[train_column],
                                                      ideal_data[ideal_column])
                    self.selection[train_column] = [ideal_column, max_dev]
        print("The following functions has been selected: \n", self.selection)
        return self.selection
