import csv
import math
import sys
from math import sqrt
from math import pi
from math import exp

# Represents the implementation of K Nearest Neighbor
class KNearestNeighbor:

    def __init__(self):
        self.data = []
        self.data_to_predict = []
        self.k = 0

    # Parses the CSV file and retrieves the names of the attributes and the data point for those attributes
    def parse_csv(self, file_name):
        has_reached_title_row = False
        with open(file_name, newline='') as csv_file:
            for row in csv.reader(csv_file):
                if not has_reached_title_row:
                    has_reached_title_row = True
                else:
                    attribute_data = []
                    for element in row:
                        attribute_data.append(float(element))
                    self.data.append(attribute_data)
        self.k = round(math.sqrt(len(self.data)))

    # Aggregates the training and test data to then compute the distance function passed as the second argument
    def run(self, data_to_predict, distance_function):
        self.data_to_predict = data_to_predict
        neighbors = []
        for i in range(len(self.data)):
            current_data = self.data[i]
            neighbors.append((distance_function(current_data, self.data_to_predict), i))
        neighbors = sorted(neighbors, key=lambda x : x[0])
        nearest_neighbors = []
        for i in range(self.k):
            current_neighbor = neighbors[i]
            nearest_neighbors.append(self.data[current_neighbor[1]][-1])
        return  max(set(nearest_neighbors), key = nearest_neighbors.count)

# Computes the Euclidean between two vectors
def euclidean_distance(v1, v2):
    difference_squared_sum = 0
    for i in range(len(v2)):
        difference = v1[i] - v2[i]
        difference_squared_sum += math.pow(difference, 2)
    return math.sqrt(difference_squared_sum)

# Represents the implementation of Naive Bayes
class NaiveBayes:

    def __init__(self):
        self.data = []
        self.data_to_predict = []
        self.number_of_rows = 0

    # Parses the CSV file and retrieves the names of the attributes and the data point for those attributes
    def parse_csv(self, file_name):
        has_reached_title_row = False
        with open(file_name, newline='') as csv_file:
            for row in csv.reader(csv_file):
                if not has_reached_title_row:
                    has_reached_title_row = True
                else:
                    self.number_of_rows = self.number_of_rows + 1
                    attribute_data = []
                    for element in row:
                        attribute_data.append(float(element))
                    self.data.append(attribute_data)

    # Computes the standard deviation between a vector of values
    def compute_standard_deviation(self, values):
        sum_val = 0
        for v in values:
            sum_val += v
        avg = sum_val / float(len(values))
        sum_val = 0
        for val in values:
            sum_val += math.pow(val - avg, 2)
        variance = sum_val / float(len(values) - 1)
        return sqrt(variance)

    # Itrerate through the data and for each column of the sub arrays, compute the mean, and standard deviation for them
    def compute_column_statistics(self, data):
        column_statistics = []
        # Zip the list of lists to compare all the same "indexes" for the list of lists together
        for entry in zip(*data):
            sum_val = 0
            for v in entry:
                sum_val += v
            mean = sum_val / float(len(entry))
            standard_deviation = self.compute_standard_deviation(entry)
            record = {
                'mean': mean,
                'standard_deviation': standard_deviation,
                'length': float(len(entry))
            }
            column_statistics.append(record)
        attributes_stats = column_statistics[: len(column_statistics) - 1]
        return attributes_stats

    # Because the classification of the data is the last element of the list we grab that last element and use it as
    # the identifier for all elements with that classification. We then append all the lists of data
    # for that classification into a hashmap
    def group_data_with_classifier(self):
        groups = {}
        for data_points in self.data:
            classifier = data_points[len(data_points) - 1]
            if classifier in groups:
                groups[classifier].append(data_points)
            else:
                groups[classifier] = []
                groups[classifier].append(data_points)
        return groups

    # Split dataset by the determining 'type's of cluster's and then calculate statistics for each column for each
    # individual cluster
    def compute_classifier_statistics(self):
        grouped_classifiers = self.group_data_with_classifier()
        classifier_col_stats = {}
        for groups in grouped_classifiers:
            classifier_col_stats[groups] = self.compute_column_statistics(grouped_classifiers[groups])
        return classifier_col_stats

    # Calculate the probabilities of predicting each class for a given column
    def run(self, data_to_predict):
        self.data_to_predict = data_to_predict
        classifier_stats = self.compute_classifier_statistics()
        group_probabilities = {}
        for group in classifier_stats:
            group_probabilities[group] = classifier_stats[group][0]['length'] / float(self.number_of_rows)
            for col in range(len(classifier_stats[group])):
                sd = classifier_stats[group][col]['standard_deviation']
                # Calculate the Gaussian probability distribution function for each attribute of the data point trying to predict
                if sd == 0:
                    group_probabilities[group] = 0
                else:
                    group_probabilities[group] *= (1 / (sqrt(2 * pi) * sd)) * exp(-(
                            math.pow(self.data_to_predict[col] - classifier_stats[group][col]['mean'], 2) / (
                                2 * math.pow(sd, 2))))
        prediction = max(group_probabilities, key=group_probabilities.get)
        return prediction

# Takes in the algorithm used and the data to predict and runs the algorithm on the test data and
# computes the Jaccard Index once the algorithms are done predicting for each row
def jaccard_index(algo, tests, kn):
    expected_values = []
    results = []
    for test in tests:
        expected_values.append(test[len(test) - 1])
        del test[len(test) - 1]
        result = 0
        if kn:
            result = algo.run(test, euclidean_distance)
        else:
            result = algo.run(test)
        results.append(result)
    intersection = 0
    for i in range(0, len(expected_values)):
        er = expected_values[i]
        res = results[i]
        if er == res:
            intersection += 1
    return intersection / (len(expected_values) + len(results) - intersection)

# Parses the test CSV files
def parse_test_csv(file_name):
    data = []
    with open(file_name, newline='') as csv_file:
        for row in csv.reader(csv_file):
                attribute_data = []
                for element in row:
                    attribute_data.append(float(element))
                data.append(attribute_data)
    return data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid usage. Valid argument is python3 main.py <test csv_file> <train csv file>")
    test_file_name = sys.argv[1]
    train_file_name = sys.argv[2]
    tests = parse_test_csv(test_file_name)
    # Put test data in this array
    naive_bayes = NaiveBayes()
    naive_bayes.parse_csv(train_file_name)
    ji = jaccard_index(naive_bayes, tests, False)
    print(f"Jacard Index for Naive Bayes is {ji}")
    k_nearest_neighbor = KNearestNeighbor()
    k_nearest_neighbor.parse_csv(train_file_name)
    tests = parse_test_csv(test_file_name)
    ji2 = jaccard_index(k_nearest_neighbor, tests, True)
    print(f"Jacard Index for K Nearest Neighbor is {ji2}")
