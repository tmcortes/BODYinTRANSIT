import csv
from collections import defaultdict

def find_accelerometers_and_indexes(csv_filepath):
    with open(csv_filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header line

    # Find all columns containing 'acceleration' (case insensitive)
    accel_columns = [(i, col) for i, col in enumerate(header) if 'acceleration' in col.lower()]

    # Group by sensor prefix (before the last '_x', '_y', '_z')
    sensors = defaultdict(dict)  # sensor_name -> {'x': idx, 'y': idx, 'z': idx}
    for idx, col in accel_columns:
        if col.endswith('_x'):
            sensor = col[:-2]
            sensors[sensor]['x'] = idx
        elif col.endswith('_y'):
            sensor = col[:-2]
            sensors[sensor]['y'] = idx
        elif col.endswith('_z'):
            sensor = col[:-2]
            sensors[sensor]['z'] = idx

    # Filter sensors that have all 3 axes and prepare the list of indexes
    accelerometer_indexes = []
    for sensor, axes_dict in sensors.items():
        if all(axis in axes_dict for axis in ['x', 'y', 'z']):
            accelerometer_indexes.append([axes_dict['x'], axes_dict['y'], axes_dict['z']])

    num_accelerometers = len(accelerometer_indexes)
    return num_accelerometers, accelerometer_indexes


# Example usage:
csv_file_path = 'p011-control-1_rokoko_data_unified.csv'  # Replace with your CSV filename
count, indexes = find_accelerometers_and_indexes(csv_file_path)

print(f"Number of accelerometers detected: {count}")
print("Indexes of accelerometer columns (x, y, z):")
print(indexes)
flist = [i - 1 for sublist in indexes for i in sublist]
print(flist)

with open('accelerometer_columns.txt', 'w') as f:
    for idx in flist:
        f.write(f"{idx}\n")