import pandas as pd
from sqlalchemy import insert
from .models import create_session, TrainData, IdealFunctions


class InsertData:
    """
    Handles the insertion of data from CSV files into the database.
    """

    def __init__(self, train_path, ideal_path):
        """
        Constructs all the necessary attributes for the InsertData object.
        """
        self.train_path = train_path
        self.ideal_path = ideal_path

    def bulk_insert(self):
        """
        Reads data from the entered CSV files and inserts it into the database.
        """
        try:
            # Reads CSV data as Pandas DataFrame
            self.train_dataset = pd.read_csv(self.train_path)
            self.ideal_dataset = pd.read_csv(self.ideal_path)
            print("train & ideal CSV files were loaded successfully.")
        except FileNotFoundError as e:
            print(f"CSV file not found: {e.filename}")
            exit()

        session = create_session()

        # Prepares data for bulk insertion
        datasets = {
            TrainData: self.train_dataset.to_dict(orient='records'),
            IdealFunctions: self.ideal_dataset.to_dict(orient='records')
        }

        # Bulk inserts data into the specified table.
        with session as local_session:
            try:
                for table, dataset in datasets.items():
                    local_session.execute(insert(table), dataset)
                local_session.commit()
                print("Data was successfully inserted into the database.")
            except Exception as e:
                # Rollback in case of error
                local_session.rollback()
                print(f"Data bulk insert failed. Error occurred: {e}")
