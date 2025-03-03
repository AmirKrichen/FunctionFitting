import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from database.database_setup import InsertData


class TestInsertData(unittest.TestCase):
    """
    Unit tests for the InsertData class.
    """
    def setUp(self):
        """
        Set up mock data for testing.
        """
        self.mock_train_data = pd.DataFrame({
            'x': [1, 2], 'y1': [10, 20], 'y2': [-10, -20],
            'y3': [10, 20], 'y4': [-10, -20]})

        self.mock_ideal_data = pd.DataFrame(
            {f"y{i}": [10, 20] for i in range(1, 51)})
        self.mock_ideal_data['x'] = [1, 2]

        self.mock_test_data = pd.DataFrame({
            'x': [1, 2], 'y': [10, 20]})

    @patch('pandas.read_csv')
    @patch('database.database_setup.create_session')
    def test_bulk_insert(self, mock_create_session, mock_read_csv):
        """
        Tests the bulk insert functionality of InsertData.
        """
        # Create a mock session object
        mock_session = MagicMock()

        # Set the session as a context manager
        mock_create_session.return_value = mock_session
        mock_create_session.return_value.__enter__.return_value = mock_session
        mock_create_session.return_value.__exit__.return_value = None

        # Mocks read_csv method to return predefined DataFrames
        mock_read_csv.side_effect = [self.mock_train_data,
                                     self.mock_ideal_data,
                                     self.mock_test_data]

        # Create an instance of InsertData with mock parameters
        data_loader = InsertData(train_path='train.csv',
                                 ideal_path='ideal.csv',
                                 test_path='test.csv')
        data_loader.bulk_insert()

        # Verify that read_csv was called three times
        self.assertTrue(mock_read_csv.called)
        self.assertEqual(mock_read_csv.call_count, 3)

        # Verify that session execute was called for each table
        self.assertEqual(mock_session.execute.call_count, 3)


if __name__ == '__main__':
    unittest.main()
