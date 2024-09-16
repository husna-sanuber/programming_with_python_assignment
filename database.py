# Import the MySQL connector
import mysql.connector

# Function to create the initial connection to MySQL server
def connect_to_mysql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=""
    )

# Function to execute individual SQL queries
def execute_query(cursor, query):
    try:
        cursor.execute(query)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        
# Function to create database if it doesn't exist
def create_database(cursor):
    # Drop the existing database if it exists
    drop_db_query = "DROP DATABASE IF EXISTS python_assignment_db"
    create_db_query = "CREATE DATABASE w"
    
    execute_query(cursor, drop_db_query)  # Drop if it exists
    execute_query(cursor, create_db_query)  # Create new database

# Function to connect to the new database
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="python_assignment_db"
    )

# Function to create all the required tables
def create_tables(cursor):
    # List of SQL queries to create tables
    table_queries = [
        "CREATE TABLE train (id INT AUTO_INCREMENT PRIMARY KEY, x FLOAT, y1 FLOAT, y2 FLOAT, y3 FLOAT, y4 FLOAT)",
        "CREATE TABLE test (id INT AUTO_INCREMENT PRIMARY KEY, x FLOAT, y FLOAT)",
        "CREATE TABLE best_fit_func (id INT AUTO_INCREMENT PRIMARY KEY, x FLOAT, y FLOAT, choosen_func FLOAT)",
        "CREATE TABLE mapping (id INT AUTO_INCREMENT PRIMARY KEY, x FLOAT, y FLOAT, ideal_x FLOAT, ideal_y FLOAT, deviation FLOAT)",
        "CREATE TABLE ideal (id INT AUTO_INCREMENT PRIMARY KEY, x FLOAT)"
    ]

    # Execute each table creation query
    for query in table_queries:
        execute_query(cursor, query)

# Function to add columns to the 'ideal' table
def add_columns_to_ideal(cursor):
    # SQL query to add 50 additional columns to the 'ideal' table
    for i in range(50):
        add_column_query = f"ALTER TABLE ideal ADD COLUMN y{i+1} FLOAT"
        execute_query(cursor, add_column_query)

# Main program execution
def main():
    # Step 1: Connect to MySQL server
    mydb = connect_to_mysql()
    mycursor = mydb.cursor()

    # Step 2: Create the database
    create_database(mycursor)

    # Step 3: Connect to the newly created database
    mydb_new = connect_to_database()
    mycursor_new = mydb_new.cursor()

    # Step 4: Create the required tables
    create_tables(mycursor_new)

    # Step 5: Add additional columns to 'ideal' table
    add_columns_to_ideal(mycursor_new)

    # Commit all changes
    mydb_new.commit()

    # Close the connections
    mycursor.close()
    mydb.close()
    mycursor_new.close()
    mydb_new.close()

    print("Database and tables have been created successfully!")

# Entry point of the script
if __name__ == "__main__":
    main()
