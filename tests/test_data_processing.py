import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from parameterized import parameterized
from ops_viz.data_processing import MathUtils, ProcessData


class TestMathUtils(unittest.TestCase):
    """
    Unit tests for the MathUtils class.
    """
    @parameterized.expand([
        # Defines different test cases to test our method
        (pd.Series([-2, -1.5, -1, 0]), pd.Series([-2, -1, -0.5, 0]), 0.5),
        (pd.Series([0, 0.5, 1, 2]), pd.Series([0, 0.5, 1.5, 2]), 0.25),
        (pd.Series([0, 0.01, -0.01]), pd.Series([0, 0.02, -0.02]), 2e-4),
        ])
    def test_sqd_dev_sum(self, column1, column2, sqd_dev_res):
        """
        Tests the squared deviation sum calculation between two columns.

        :param sqd_dev_res: Expected squared deviation sum result
        """
        math_utils = MathUtils()
        self.assertEqual(math_utils.sqd_dev_sum(column1, column2), sqd_dev_res)

    @parameterized.expand([
        # Defines different test cases to test our method
        (pd.Series([-2, -1.5, -1, 0]), pd.Series([-2, -1, -0.5, 0]), 0.5),
        (pd.Series([0, 0.5, 1, 2]), pd.Series([0, 0.5, 1.5, 2]), 0.5),
        (pd.Series([0, 0.01, -0.01]), pd.Series([0, 0.02, -0.02]), 0.01),
        ])
    def test_max_deviation(self, column1, column2, max_dev):
        """
        Tests the maximum deviation calculation between two columns.

        :param max_dev: Expected maximum deviation result
        """
        math_utils = MathUtils()
        self.assertEqual(math_utils.max_deviation(column1, column2), max_dev)


class TestProcessData(unittest.TestCase):
    """
    Unit tests for the ProcessData class.
    """
    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Defines mock train, ideal functions and test tables
        self.mock_train_data = pd.DataFrame({
            'x': [-0.1, 0, 0.1, 0.2,],
            'y1': [1, 2, 3, 4],
            'y2': [-10, -20, -30, -40],
        })
        self.mock_ideal_data = pd.DataFrame({
            'x': [-0.1, 0, 0.1, 0.2,],
            'y9': [100, 200, 300, 400],
            'y10': [-100, -200, -300, -400],
            'y11': [2, 3, 4, 5],
            'y12': [-11, -21, -21, -41],
        })
        self.mock_test_data = pd.DataFrame({
            'x': [-0.1, 0, 0, 0.2,],
            'y': [1, -19, 2, -39]
        })
        # Defines the expected selected function for the test
        self.mock_selection = {'y1': ['y11', 1], 'y2': ['y12', 9]}

    @patch('ops_viz.data_processing.DataHandler.get_data')
    def test_select_functions(self, mock_get_data):
        """
        tests that ProcessData correctly selects functions based on
        mocked train and ideal data.
        """
        # Mocks train and ideal functions tables
        mock_get_data.side_effect = [self.mock_ideal_data,
                                     self.mock_train_data]
        # Creates an instance of ProcessData with mock parameters
        data_processor = ProcessData(session=None)
        result = data_processor.select_functions()
        # Verifies the result
        self.assertEqual(result, self.mock_selection)

    @patch('pandas.DataFrame.to_sql')
    @patch('ops_viz.data_processing.DataHandler.get_data')
    def test_insert_test_data(self, mock_get_data, mock_to_sql):
        """
        Ensures that ProcessData updates test data correctly and
        performs a database insertion
        """
        # Mocks test and ideal functions tables
        mock_get_data.side_effect = [self.mock_test_data, self.mock_ideal_data]

        # Defines the expected Dataframe after mapping.
        expected_test_data = self.mock_test_data.copy()
        expected_test_data['delta_y'] = [1, 2, 1, 2]
        expected_test_data['ideal_function'] = ['y11', 'y12', 'y11', 'y12']

        # Mocks a session object
        mock_session = MagicMock()
        mock_session.bind = MagicMock()

        # Creates an instance of ProcessData with mock parameters
        data_processor = ProcessData(session=mock_session)
        data_processor.selection = self.mock_selection
        data_processor.insert_test_data()

        # Verifies if self.mock_test_data is updated to match expected result
        self.assertEqual(self.mock_test_data.to_dict(),
                         expected_test_data.to_dict())
        # Verifies if the database insertion was attempted
        mock_to_sql.assert_called_once()


if __name__ == '__main__':
    unittest.main()
