from database.models import create_session
from database.database_setup import InsertData
from ops_viz.data_processing import ProcessData
from ops_viz.visualizations import VisualizeData


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

    # Visualize results
    data_visualizer = VisualizeData(functions=selected_functions,
                                    session=session)
    # Compare training data with ideal functions to see how they align.
    data_visualizer.plot_train_vs_ideal()
    # Show how test data aligns or deviates from each ideal function.
    data_visualizer.plot_test_vs_ideal()
    # Overlay test data on ideal functions to visualize the mapping we did.
    data_visualizer.plot_test_over_ideal()


if __name__ == "__main__":
    main()
