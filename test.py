import arff

# Load the ARFF file 
data = arff.load(open('2017.arff', 'r'))

# Print the attributes (column headers)
print(data['attributes'])  

# Print the data
print(data['data'])