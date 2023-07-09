import os

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a relative path to the data file
data_path = os.path.join(script_dir, 'data', 'data.txt')
print(data_path)