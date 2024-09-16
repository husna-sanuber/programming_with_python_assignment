import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import mysql.connector
from mysql.connector import Error
import sys
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool

# Database connection
def connect_to_database():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="python_assignment_db"
        )
    except Error as e:
        print(f"Error: {e}")
        sys.exit()

# Check if the database exists
def check_database(cursor):
    cursor.execute("SHOW DATABASES")
    databases = cursor.fetchall()
    database_list = [db[0] for db in databases]
    
    if 'python_assignment_db' not in database_list:
        print('Database not found. Please run the setup script first!')
        sys.exit()

# Load CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    x_vals = data['x'].values
    y_vals = data[[col for col in data.columns if col.startswith('y')]].values
    return x_vals, y_vals

# Ideal function for fitting
def ideal_function(x, a, b):
    return a * x + b

# Fit ideal functions to training data and find the best fit
def calculate_best_fit(train_data, ideal_funcs):
    best_fit_parameters = []
    lowest_deviation = float('inf')
    all_min_deviations = []
    x_values, y_values = train_data

    index = 0
    for y_train_values in y_values:
        if index < 4:  # Ensure we only fit the first 4 columns
            for ideal_y_values in ideal_funcs:
                # Fit the curve for the current column of y_values
                fit_params, _ = curve_fit(ideal_function, x_values, y_values[:, index])
                predicted_y = ideal_function(x_values, *fit_params)
                
                # Calculate the sum of squared deviations
                squared_deviation_sum = np.sum((y_values[:, index] - predicted_y) ** 2)

                if squared_deviation_sum < lowest_deviation:
                    lowest_deviation = squared_deviation_sum
                    best_fit = fit_params
            
            # Move to the next column of y_values
            index += 1
            best_fit_parameters.append(best_fit)
            all_min_deviations.append(lowest_deviation)
    
    return best_fit_parameters, all_min_deviations

# Map test data to the chosen ideal functions
def assign_test_data(test_data, selected_functions):
    x_values_test, y_values_test = test_data
    test_mappings = []

    for x_value, y_value in zip(x_values_test, y_values_test):
        all_deviations = []
        for function_params in selected_functions:
            predicted_y = ideal_function(x_value, *function_params)
            deviation = np.abs(y_value - predicted_y)
            all_deviations.append(deviation)
        
        smallest_deviation = min(all_deviations)
        if smallest_deviation < np.sqrt(2) * np.max(all_deviations):
            best_fit_idx = all_deviations.index(smallest_deviation)
            best_fit_params = selected_functions[best_fit_idx]
            test_mappings.append((x_value, y_value, best_fit_params, smallest_deviation))
        else:
            test_mappings.append((x_value, y_value, None, None))
    
    return test_mappings

# Helper function to convert numpy types to native Python types
def convert_to_native_types(data):
    if isinstance(data, (list, tuple)):
        return [convert_to_native_types(item) for item in data]
    else:
        if isinstance(data, np.float64) or isinstance(data, np.float32):
            return float(data)
        elif isinstance(data, np.int64) or isinstance(data, np.int32):
            return int(data)
        else:
            return data

# Insert data into the MySQL database
def insert_data_to_db(cursor, db_connection, query, data_list):
    try:
        native_data = convert_to_native_types(data_list)
        cursor.executemany(query, native_data)
        db_connection.commit()
    except Error as e:
        print(f"Database insertion error: {e}")

# Plot the data using Bokeh
def plot_data(x_train, y_train, x_test, y_test, chosen_functions, mappings):
    p = figure(title="Training Data and Ideal Functions", x_axis_label="x", y_axis_label="y")
    
    # Plot training data
    for i in range(y_train.shape[1]):
        p.scatter(x_train, y_train[:, i], legend_label=f'Training y{i+1}', color='blue', size=6)
    
    # Plot ideal functions
    x_range = np.linspace(min(x_train), max(x_train), 500)
    for i, params in enumerate(chosen_functions):
        y_range = ideal_function(x_range, *params)
        p.line(x_range, y_range, legend_label=f'Ideal Function {i+1}', color='green')
    
    # Plot test data and mappings
    mapped_x = [x for x, y, func, dev in mappings if func is not None]
    mapped_y = [y for x, y, func, dev in mappings if func is not None]
    p.scatter(mapped_x, mapped_y, color='red', legend_label="Mapped Test Data", size=10)

    # Add hover tool for better interactivity
    source = ColumnDataSource(data=dict(x=mapped_x, y=mapped_y))
    hover = HoverTool(tooltips=[("x", "@x"), ("y", "@y")])
    p.add_tools(hover)
    
    p.legend.location = "top_left"
    output_file("data_plot.html")
    show(p)

# Main function to handle the flow
def main():
    # Step 1: Database connection
    db_connection = connect_to_database()
    mycursor = db_connection.cursor()

    # Step 2: Check if the required database exists
    check_database(mycursor)
    
    # Step 3: Load CSV files
    x_train, y_train = load_data("train.csv")
    x_test, y_test = load_data("test.csv")
    ideal_data = load_data("ideal.csv")
    
    # Step 4: Insert training, test, and ideal data into MySQL database
    train_data = [(x, y1, y2, y3, y4) for x, y1, y2, y3, y4 in zip(x_train, y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3])]
    insert_data_to_db(mycursor, db_connection, 
                      "INSERT INTO train (x, y1, y2, y3, y4) VALUES (%s, %s, %s, %s, %s)", 
                      train_data)
    
    test_data = [(x, y) for x, y in zip(x_test, y_test[:, 0])]
    insert_data_to_db(mycursor, db_connection, 
                      "INSERT INTO test (x, y) VALUES (%s, %s)", 
                      test_data)
    
    # Ensure that ideal_data has 51 columns (x + y1 to y50)
    ideal_data_list = [(x, *y_row) for x, y_row in zip(ideal_data[0], ideal_data[1])]
    insert_data_to_db(mycursor, db_connection, 
                      "INSERT INTO ideal (x, " + ", ".join([f"y{i+1}" for i in range(50)]) + ") VALUES (" + ", ".join(["%s"] * 51) + ")", 
                      ideal_data_list)
    
    # Step 5: Fit ideal functions and map test data
    chosen_functions, deviations = calculate_best_fit((x_train, y_train), ideal_data)
    mappings = assign_test_data((x_test, y_test), chosen_functions)
    
    # Step 6: Insert best fit functions into MySQL database
    best_fit_data = [(float(params[0]), float(params[1]), float(dev)) for params, dev in zip(chosen_functions, deviations)]
    insert_data_to_db(mycursor, db_connection, 
                      "INSERT INTO best_fit_func (x, y, choosen_func) VALUES (%s, %s, %s)", 
                      best_fit_data)
    
    # Step 7: Insert mappings into MySQL database
    mapping_data = []
    for x, y, best_fit_params, deviation in mappings:
        if best_fit_params is not None:
            mapping_data.append((float(x), float(y), float(best_fit_params[0]), float(best_fit_params[1]), float(deviation)))
    insert_data_to_db(mycursor, db_connection, 
                      "INSERT INTO mapping (x, y, ideal_x, ideal_y, deviation) VALUES (%s, %s, %s, %s, %s)", 
                      mapping_data)

    # Step 8: Plot the data
    plot_data(x_train, y_train, x_test, y_test, chosen_functions, mappings)

    print("Extracted Ideal Function Data:")
    print("Index________x_val____________________y_val________________selected_value")
    for idx, parameters in enumerate(chosen_functions[0]):
        print(f" {idx+1}: {parameters[0]}, {parameters[1]}, {chosen_functions[1][idx]}")
        values_to_insert = [(parameters[0], parameters[1], chosen_functions[1][idx])]
        mycursor.executemany(
            "INSERT INTO best_fit_function (x_val, y_val, selected_function) VALUES (%s, %s, %s)", values_to_insert)
        db_connection.commit()

    print()
    print("Mapped Functions:")
    print()
    print("_x_val__|___y_val__|__fit_func_A__|__fit_func_B__|__Deviation")
    for idx, (x_val, y_val, best_model_params, deviation_value) in enumerate(mappings):
        if best_model_params is not None:
            print(x_val, y_val[0], best_model_params[0], best_model_params[1], deviation_value[0])
            values_to_insert = [(x_val, y_val[0], best_model_params[0],
                                best_model_params[1], deviation_value[0])]
            mycursor.executemany(
                "INSERT INTO mapping_results (x_val, y_val, ideal_x_val, ideal_y_val, deviation_value) VALUES (%s, %s, %s, %s, %s)", values_to_insert)
        else:
            print(f"Data point {idx+1}: x_val={x_val}, y_val={y_val}, No best fit found within the deviation threshold")
        db_connection.commit()
    # Close database connections
    mycursor.close()
    db_connection.close()

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Execute the main function
if __name__ == "__main__":
    main()
