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

# Function to transpose a matrix
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Function to multiply two matrices
def matrix_multiply(a, b):
    return [[sum(ai * bj for ai, bj in zip(a_row, b_col)) for b_col in transpose(b)] for a_row in a]

# Function to subtract two matrices
def matrix_subtract(a, b):
    return [[ai - bi for ai, bi in zip(a_row, b_row)] for a_row, b_row in zip(a, b)]

# Function to calculate the mean of a list
def mean(values):
    if isinstance(values, list):
        return sum(values) / len(values) if len(values) > 0 else 0
    else:
        return values


# Center the data
mean_values = [mean(col) for col in transpose(numeric_data)]
centered_data = matrix_subtract(numeric_data, [[m] * len(numeric_data) for m in mean_values])

# Function to transpose and calculate the covariance matrix
def covariance_matrix(data):
    transposed_data = transpose(data)
    return [[sum((a - mean(a)) * (b - mean(b)) for a, b in zip(row_a, row_b)) / (len(row_a) - 1) for row_b in transposed_data] for row_a in transposed_data]

# Calculate the covariance matrix
covariance_matrix_data = covariance_matrix(centered_data)

# Function to calculate the dot product of two matrices
def dot_product(a, b):
    if a is None or b is None:
        print("Warning: Unsupported type for dot product. Returning None.")
        return None
    elif isinstance(a[0], list) and isinstance(b[0], list):
        return sum(ai[0] * bi[0] for ai, bi in zip(a, b))
    elif isinstance(a[0], (int, float)) and isinstance(b[0], (int, float)):
        return sum(ai * bi for ai, bi in zip(a, b))
    else:
        print("Warning: Unsupported type for dot product. Returning None.")
        return None



# Function to perform eigendecomposition
def eigendecomposition(matrix):
    n = len(matrix)
    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Initialize a random vector
    b = [[1] for _ in range(n)]

    # Perform power iteration
    for _ in range(50):
        # Multiply matrix by vector
        Ab = matrix_multiply(matrix, b)

        # Normalize the vector, handling the case where norm is zero
        norm = max(Ab[0])
        b = [[ai / norm] if norm != 0 else [ai] for ai in Ab[0]]

    # Calculate eigenvalue
    eigenvalue = dot_product(transpose(b)[0], matrix_multiply(matrix, b))[0][0]

    # Calculate eigenvector
    eigenvector = matrix_multiply(matrix_subtract(identity_matrix, matrix_multiply(b, transpose(b))), b)

    return eigenvalue, eigenvector



# Perform eigendecomposition
eigenvalues, eigenvectors = zip(*[eigendecomposition(covariance_matrix_data) for _ in range(len(covariance_matrix_data))])

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
eigenvalues = [eigenvalues[i] for i in sorted_indices]
eigenvectors = [eigenvectors[i] for i in sorted_indices]

# Select the top k eigenvectors based on the number of components desired (k)
k = 2  # You can adjust the number of components as needed
selected_eigenvectors = eigenvectors[:k]

# Project the centered data onto the selected eigenvectors
pca_result = matrix_multiply(centered_data, transpose(selected_eigenvectors))

# Print the result of PCA
print("\nPCA Result:")
print(pca_result)
