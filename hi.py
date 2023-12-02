import numpy as np

def parse_arff(file_path):
    data_started = False
    attributes = []
    data = []
    nominal_mapping = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith('%'):
                continue  # Skip comments and empty lines

            if line.lower().startswith('@relation'):
                continue  # Skip relation information

            if line.lower().startswith('@attribute'):
                parts = line.split()
                attr_name = parts[1].strip()

                if '{' in line:  # Nominal data
                    values = line[line.index('{') + 1:line.index('}')].split(',')
                    attributes.append((attr_name, 'nominal', values))
                    # Sample attribute information
                    attribute_info = ((attr_name, 'nominal', values))
                    
                    # Extract relevant information
                    attr_name, attr_type, attr_values = attribute_info

                    #Create a mapping dictionary
                    nominal_mapping.append( {value: index for index, value in enumerate(attr_values)} )


                else:  # Numeric data
                    attributes.append((attr_name, 'numeric', 0))

            if line.lower().startswith('@data'):
                data_started = True
                continue

            if data_started:
                data_line = line.split(',')
                
                data.append(line.split(','))

    return attributes, data

file_path = r'2017.arff'
attributes, data = parse_arff(file_path)

for row in data :
    for i in range ( len (attributes) ):
        attr_name, attr_type, attr_values = attributes[i]
        if attr_type == 'nominal' :
            # Create a mapping dictionary
            nominal_mapping = {value: index for index, value in enumerate(attr_values)}
            row[i] = nominal_mapping.get(row[i])
        elif attr_type == 'numeric' :
            try :
                row[i] = float(row[i])
            except Exception as e:
                # Handle the exception
                row[i] = 1


# Extracting the numeric data for PCA
numeric_data = []
for row in data:
    numeric_row = [float(value) if attributes[i][1] == 'numeric' else row[i] for i, value in enumerate(row)]
    numeric_data.append(numeric_row)

# Convert the numeric data to a NumPy array
numeric_data_array = np.array(numeric_data)

# Center the data
mean_values = np.mean(numeric_data_array, axis=0)
centered_data = numeric_data_array - mean_values

# Calculate the covariance matrix
covariance_matrix = np.cov(centered_data, rowvar=False)

# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Select the top k eigenvectors based on the number of components desired (k)
k = 2  # You can adjust the number of components as needed
selected_eigenvectors = eigenvectors[:, :k]

# Project the centered data onto the selected eigenvectors
pca_result = np.dot(centered_data, selected_eigenvectors)

# Print the result of PCA
print("\nPCA Result:")
print(pca_result)
