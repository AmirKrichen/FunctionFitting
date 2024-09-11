from database.models import create_session
from database.database_setup import InsertData
from ops_viz.data_processing import ProcessData


def main():
    """
    Main function to orchestrate data loading, processing, and visualization.
    """
    # Resets database and create a session instance
    session = create_session(database_reset=True)

    # Inserts train and ideal data into the database
    data_loader = InsertData(train_path="./data/train.csv",
                             ideal_path="./data/ideal.csv")
    data_loader.bulk_insert()

    # Processes and analyses the data
    data_processor = ProcessData(test_path="./data/test.csv",
                                 session=session)
    # Assigns and ideal functions to each train Function (least square)
    selected_functions = data_processor.select_functions()
    # Maps individual test Data to one of the four selected ideal Functions
    data_processor.insert_test_data()


if __name__ == "__main__":
    main()
