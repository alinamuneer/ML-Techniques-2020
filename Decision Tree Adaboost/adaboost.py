from matplotlib import pyplot as plt
import sys
import math


class Point:
    def __init__(self, x, y, label):
        self.x = float(x)
        self.y = float(y)
        self.label = float(label)
        self.predicted_label = 0.0

    def __eq__(self, other):
        return self.x == other.x \
            and self.y == other.y \
            and self.label == other.label

    def __hash__(self):
        return hash((self.x, self.y, self.label))

    def __str__(self):
        return f"({self.x}, {self.y})"

    @staticmethod
    def load(path):
        points = []
        with open(path, 'r') as file:
            for line in file:
                points.append(Point(*line.split('\t')))
        return points


class InvalidClassifierError(Exception):
    pass


class Classifier:
    def __init__(self, value, sign):
        self.value = value
        self.sign = sign
        self.alpha = 0

    def flip_sign(self):
        self.sign = not self.sign

    def classify(self, point):
        raise NotImplementedError


class HorizontalClassifier(Classifier):
    def __init__(self, value, classify_above_as_true):
        super().__init__(value, classify_above_as_true)

    def classify(self, point):
        above = point.y > self.value

        if self.sign:
            return 1.0 if above else -1.0
        else:
            return -1.0 if above else 1.0


class VerticalClassifier(Classifier):
    def __init__(self, value, classify_left_as_true):
        super().__init__(value, classify_left_as_true)

    def classify(self, point):
        left = point.x < self.value

        if self.sign:
            return 1.0 if left else -1.0
        else:
            return -1.0 if left else 1.0


def main():
    try:
        iterations = int(sys.argv[1])
    except IndexError:
        iterations = 100

    print(f"Using {iterations} iterations")

    n_classifiers = iterations * 2

    points = Point.load('data.txt') #using data.txt because it has no /t before every example unlike dataCircle.txt

    positives = list(filter(lambda point: point.label == 1.0, points))
    negatives = list(filter(lambda point: point.label == -1.0, points))

    plt.subplot(211)

    plt.plot([point.x for point in positives], [point.y for point in positives], 'go')
    plt.plot([point.x for point in negatives], [point.y for point in negatives], 'ro')

    #plt.show()

    classifiers = initialize_classifiers(points, n_classifiers)

    weights = {}
    for point in points:
        weights[point] = 1 / len(points)

    best_classifiers = []
    for i in range(iterations):
        error, best_classifier = select_classifier_with_min_error(weights, classifiers)
        if error < 0.5:
            best_classifiers.append(best_classifier)
        else:
            continue

        best_classifier.alpha = 0.5 * math.log((1 - error) / error)
        Z = normalization_factor(weights, best_classifier)

        for point in weights:
            weights[point] = 1/Z * weights[point] * \
                math.exp(-best_classifier.alpha * point.label * best_classifier.classify(point))

        classification_error = total_classification_error(points, best_classifiers)
        if classification_error == 0:
            print(f"Reached an error of 0, breaking early after {i + 1} iterations")
            break

    final_error = total_classification_error(points, best_classifiers)
    correct_classifications = len(points) - final_error

    print(f"Final number of classifiers: {len(best_classifiers)}")
    print(f"Correctly classified {correct_classifications} out of {len(points)} points")
    p = round(100.0 * correct_classifications / len(points), 2)
    print(f"That's a percentage of {p}")

    positives = list(filter(lambda point: point.predicted_label == 1.0, points))
    negatives = list(filter(lambda point: point.predicted_label == -1.0, points))

    plt.subplot(212)

    plt.plot([point.x for point in positives], [point.y for point in positives], 'go')
    plt.plot([point.x for point in negatives], [point.y for point in negatives], 'ro')

    for classifier in best_classifiers:
        if isinstance(classifier, VerticalClassifier):
            plt.plot([classifier.value, classifier.value], [-10.0, 10.0], 'k-' if classifier.sign else 'b-')
        elif isinstance(classifier, HorizontalClassifier):
            plt.plot([-10.0, 10.0], [classifier.value, classifier.value], 'k-' if classifier.sign else 'b-')
        else:
            raise InvalidClassifierError

    plt.show()


def total_classification_error(points, classifiers):
    incorrect_classifications = 0
    for point in points:
        prediction = 0
        for classifier in classifiers:
            prediction += classifier.alpha * classifier.classify(point)

        point.predicted_label = 1.0 if prediction > 0.0 else -1.0
        if point.predicted_label != point.label:
            incorrect_classifications += 1

    return incorrect_classifications


def normalization_factor(weights, classifier):
    Z = 0
    for point in weights:
        Z += weights[point] * math.exp(-classifier.alpha * point.label * classifier.classify(point))
    return Z


def select_classifier_with_min_error(weights, classifiers):
    min_error = 1
    best_classifier = None

    for classifier in classifiers:
        error = classifier_error(weights, classifier)
        if error < min_error:
            min_error = error
            best_classifier = classifier

    return min_error, best_classifier


def classifier_error(weights, classifier):
    error = 0
    for point in weights:
        if classifier.classify(point) != point.label:
            error += weights[point]

    if error > 0.5:
        classifier.flip_sign()
        error = 1 - error
    return error


def initialize_classifiers(points, n_classifiers):
    classifiers = []
    points_min_value_x = min(point.x for point in points)
    points_min_value_y = min(point.y for point in points)

    points_max_value_x = max(point.x for point in points)
    points_max_value_y = max(point.y for point in points)

    step_vertical = abs(points_max_value_x - points_min_value_x) / (n_classifiers // 2)
    step_horizontal = abs(points_max_value_y - points_min_value_y) / (n_classifiers // 2)

    horizontal_value = points_min_value_y
    vertical_value = points_min_value_y

    # make grid of horizontal and vertical classifiers
    for i in range(n_classifiers // 2):
        classifiers.append(HorizontalClassifier(horizontal_value, True))
        classifiers.append(VerticalClassifier(vertical_value, True))

        horizontal_value += step_horizontal
        vertical_value += step_vertical

    return classifiers


if __name__ == "__main__":
    main()

