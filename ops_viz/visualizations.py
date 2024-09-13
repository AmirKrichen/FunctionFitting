import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from .data_processing import DataHandler


class VisualizeData(DataHandler):
    """
    Handles the visualization of training, test, and ideal function data.

    This class provides methods to create different types of plots to:
        - Compare training data with ideal functions to see how they align.
        - Show how test data aligns or deviates from each ideal function.
        - Overlay test data on ideal functions to visualize the mapping we did.
        - Create individual plots for test data against each ideal function.
    """
    def __init__(self, functions, session):
        super().__init__(session)
        self.functions = functions
        self.train_data = self.get_data('train_data')
        self.ideal_data = self.get_data('ideal_functions')
        self.test_data = self.get_data('test_data')

    def plot_train_vs_ideal(self):
        """
        Plots training data against ideal functions across multiple subplots.

        Saves the plot in output folder an PNG image file.
        """
        # Defines layout and create figure with subplots
        layout = 'constrained'
        panels = [['y1 panel', 'y2 panel'], ['y3 panel', 'y4 panel']]
        fig, axs = plt.subplot_mosaic(panels, figsize=(9, 9), layout=layout)

        # Sets the main title for the entire figure
        fig.suptitle("Comparison of Train Data with selected Ideal Functions")

        # Iterates over each subplot and plot the data
        for i, ax in enumerate(axs.values()):
            # Retrieves the ideal function for the corresponding train function
            current_y = self.functions[f'y{i+1}'][0]

            # Sets the title for the current subplot
            ax.set_title(f'Train y{i+1} and Ideal {current_y}',
                         fontsize='medium')
            # Plots the training data line
            ax.plot(self.train_data['x'],
                    self.train_data[f'y{i+1}'],
                    label=f'Train y{i+1}')
            # Plots the ideal data line
            ax.plot(self.ideal_data['x'],
                    self.ideal_data[current_y],
                    label=f'Ideal {current_y}')

            # Sets axis labels, grid, and legend
            ax.set_xlabel('X value')
            ax.set_ylabel('Y value')
            ax.grid(alpha=0.4)
            ax.legend(fontsize='small')

        try:
            # Displays the plot and saves it to to output folder
            plt.savefig('Output/train_vs_ideal.png')
            print('figure was saved successfully to output folder.')
            plt.show()
            plt.close()
        except PermissionError:
            print("Error: Permission denied when trying to save the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def plot_test_vs_ideal(self):
        """
        Plots test data against ideal functions and showcases deviation region.

        Saves the plot in output folder an PNG image file.
        """
        # Defines layout and create figure with subplots
        layout = 'constrained'
        panels = [['y1 panel', 'y2 panel'], ['y3 panel', 'y4 panel']]
        fig, axs = plt.subplot_mosaic(panels, figsize=(9, 9), layout=layout)

        # Sets the main title for the entire figure
        fig.suptitle("Test Data vs. Ideal Functions: Deviation Analysis")
        # Retrieves ideal_function column from test data table
        mapped_ideal_test = self.test_data['ideal_function']

        # Iterates over each subplot and plot the data
        for i, ax in enumerate(axs.values()):
            # Retrieves the current ideal function and its deviation
            current_y = self.functions[f'y{i+1}'][0]
            current_y_column = self.ideal_data[current_y]
            ideal_max_dev = self.functions[f'y{i+1}'][1]

            # Sets the title for the current subplot
            ax.set_title(f'Test data and Ideal {current_y}',
                         fontsize='medium')

            # Plots the selected ideal function line
            ax.plot(self.ideal_data['x'],
                    self.ideal_data[current_y],
                    label=f'Ideal function {current_y}',
                    zorder=1.2)

            # Filter mapped and unmapped test values for plotting
            is_mapped = mapped_ideal_test == current_y
            mapped_test = self.test_data[is_mapped]
            unmapped_test = self.test_data[~is_mapped]

            # Plots unmapped test values corresponding to each ideal function
            ax.scatter(unmapped_test['x'],
                       unmapped_test['y'],
                       label='Unmapped test values',
                       color='grey', s=20, zorder=1.1)

            # Plots mapped test values corresponding to each ideal function
            ax.scatter(mapped_test['x'],
                       mapped_test['y'],
                       label='Mapped test values',
                       color='red', s=20, zorder=1.1)

            # Retrieves Deviation threshold Intervals
            ymin = current_y_column - ideal_max_dev * np.sqrt(2)
            ymax = current_y_column + ideal_max_dev * np.sqrt(2)
            # Plots the Root-Mean-Square Threshold region of the deviation
            ax.fill_between(self.ideal_data['x'],
                            ymin,
                            ymax,
                            alpha=0.25, color='grey', zorder=1,
                            label='Threshold region')

            # Sets axis labels and legend
            ax.set_xlabel('X value')
            ax.set_ylabel('Y value')
            ax.legend(fontsize='xx-small')

        try:
            # Displays the plot and saves it to to output folder
            plt.savefig('Output/test_vs_ideal.png')
            print('figure was saved successfully to output folder.')
            plt.show()
            plt.close()
        except PermissionError:
            print("Error: Permission denied when trying to save the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
