import math
vtime = 10
runtime = 10
sigmoid = 1/(1+math.exp(-(vtime/runtime)))
# Import the math module


# Define the sigmoid function
def sigmoid(x):
    # Calculate the sigmoid of x
    return 1 / (1 + math.exp(-x))

print(sigmoid(0.3))