### Neural network project

Created for Dr. Baroody's AP computer science final project :)

### Structure

I have multiple classes with a few base class,
The linearlayer and convolution classes are the base layer classes,
The FF and CNN classes are abstracted collections of linearlayer and convolution which makes it easier to run them.
The loss class and MSE and huber have a is a relationship, as loss defines some functions a loss function must have.
There are also a few helper classes that when given an array, creates a list of linearlayers that fits the array you give it.
There are multiple test files, in main, there is a basic FF layer learning test,
in sine_test, there is a basic sine wave approximation test,
in MNIST, there exists a model that can be fitted onto the MNIST dataset.