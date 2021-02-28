import numpy as np
import csv

def import_data():
    X = np.genfromtxt("train_X_lr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", dtype=np.float64, delimiter=',')
    return X,Y

def compute_cost(X, Y, W):
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y                        
    squared_difference = np.square(difference)      
    mse = np.sum(squared_difference)
    cost_value = mse/(2 * len(X))
    return cost_value

def compute_gradient_of_cost_function(X, Y, W):
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y
    dW = (1/len(X)) * (np.dot(difference.T, X))
    dW = dW.T
    return dW

def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    for i in range(num_iterations):
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - (learning_rate * dW)
        cost = compute_cost(X,Y,W)
        if(i%10000==0):
            print(i,cost)
    return W

def train_model(X,Y):
    X = np.insert(X, 0, 1, axis=1)
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1],1))
    W = optimize_weights_using_gradient_descent(X,Y,W,100000000,0.0001)
    return W

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X,Y = import_data()
    weights = train_model(X,Y)
    save_model(weights, "WEIGHTS_FILE.csv")