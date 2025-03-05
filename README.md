# Function Fitting Program

## Description
This program is designed to fit test data to the closest ideal functions based on training data using least squares deviation. It involves database operations, data processing, visualization, and unit testing. 

The project uses SQLite for database management, SQLAlchemy for ORM, pandas for data manipulation, and matplotlib for visualization.

## Features
- **Database Setup**:
  - Automatically creates and populates tables for training, ideal, and test datasets.
- **Data Processing**: 
  - Identifies best-fit ideal functions using least squares method.
  - Maps test data points to the selected ideal functions with deviation threshold (√2 * max_deviation).
- **Visualization**:
  - Generates plots comparing training data with ideal functions.
  - Visualizes test data mapping, deviations, and residual errors.
  - Saves plots as PNG files in  `Output` folder.
- **Unit Tests**:
  - Validates database operations and mathematical calculations.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AmirKrichen/FunctionFitting
2. **Change directory**:
   ```bash   
   cd FunctionFitting
3. **Install dependencies**:
   ```bash   
   py -m pip install -r requirements.txt


## Project Structure
   ```bash
.
├── data/                            # Input CSV files
│   ├── ideal.csv
│   ├── test.csv
│   └── train.csv
│
├── database/
│   ├── models.py                    # Database ORM models
│   └── database_setup.py            # Data insertion logic
│
├── ops_viz/
│   ├── data_processing.py           # Analysis algorithms
│   └── visualizations.py            # Plot generation
│
├── tests/
│   ├── test_database_setup.py       # Database insertion unit tests
│   └── test_data_processing.py      # Algorithm validation tests
│
├── Output/                          # Generated PNGs visualization
│
├── main.py                          # main script
├── database.db                      # Generated database
└── requirements.txt
```

## Usage
1. **Initial Step**
    - Replace CSV files in `data` folder

2. **Testing**
    - Run unit tests to test functionality:
   ```bash
   py -m unittest discover
3. **Run the main script**
  ```bash
  python main.py
  ```
  - This will:
    - Insert data from CSV files into the database.
    - Process the data to map test points to ideal functions.
    - Generate and save visualizations in the Output folder.

## Notes
  - Ensure write permissions for `Output` folder before running the main script.
  - The database (database.db) is automatically reset when running main.py.
  - Deviation thresholds are calculated as ideal_max_dev * sqrt(2).

## Author & Course
- Developed by: Amir Krichen
- Course: DLMDSPWP01 – Programming with Python
