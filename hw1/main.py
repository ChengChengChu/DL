import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11) :
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


class NN:
    def __init__(self, layers=[2, 3, 1], activations=['sigmoid', 'sigmoid'], lr=0.01):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.lr = lr

        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))

    def forward(self, x):
        a = np.copy(x)
        z_s = []
        a_s = [a]

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_s.append(z)
            a = sigmoid(z)
            a_s.append(a)

        return z_s, a_s

    def backward(self, y, z_s, a_s):
        batch_size = y.shape[1]
        deltas = [None] * len(self.weights)
        deltas[-1] = (a_s[-1] - y) * sigmoid_derivative(z_s[-1])

        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * sigmoid_derivative(z_s[i])

        db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]

        return dw, db

    def train(self, x, y, batch_size=2, epochs=10000, lr=0.01):
        for e in range(epochs):
            i = 0
            epoch_loss = 0
            while i < len(y):
                # import pdb
                # pdb.set_trace()
                x_batch = x[i:i+batch_size].T
                y_batch = y[i:i+batch_size].T
                i = i + batch_size
                z_s, a_s = self.forward(x_batch)
                dw, db = self.backward(y_batch, z_s, a_s)
                self.weights = [w - lr * dweight for w, dweight in zip(self.weights, dw)]
                self.biases = [w - lr * dbias for w, dbias in zip(self.biases, db)]
                # import pdb
                # pdb.set_trace()
                epoch_loss += np.linalg.norm(a_s[-1] - y_batch)
            if e % 100 == 0: 
                print(f'epoch {e} loss : {epoch_loss}')

        return
    
    
    
    def test(self, x, y) :
        loss = 0.0
        num_correct = 0
        # pred_x = []
        pred_y = []
        for i in range(len(x)):
            tmp = x[i:i+1].T 
            z_s, a_s = self.forward(tmp)
            prediction = a_s[-1][0][0]
            loss += np.linalg.norm(a_s[-1] - y[i])
            predict_label = 0
            if prediction >= 0.5 : 
                predict_label = 1
            if predict_label == y[i] :
                num_correct += 1
            
            pred_y.append(predict_label)
            
            print(f"Iter {i} |  Ground truth: {y[i]} |  Prediction: {prediction}.")
        print(f"loss={loss / len(x)} accuracy={100 * num_correct / len(x)}%")

        ### Graph ###
        # plt.figure(figsize=(8, 5))
        points_x = []
        points_y = []
        colors_pred = []
        colors_label = []
        for i in range(len(x)) :
            points_x.append(x[i][0])
            points_y.append(x[i][1])

            if pred_y[i] == 0 :
                colors_pred.append('red')
            else:
                colors_pred.append('black')
            # import pdb
            # pdb.set_trace()
            if y[i][0] == 0 :
                colors_label.append('red')
            else:
                colors_label.append('black')

        # import pdb
        # pdb.set_trace()
        plt.subplot(1, 2, 1)
        plt.scatter(points_x, points_y, c=colors_label, label='truth')
        plt.title("Truth")
        
        plt.subplot(1, 2, 2)
        plt.scatter(points_x, points_y, c=colors_pred, label='prediction')
        plt.title("Prediction")
        plt.show()


        



# Example usage:
if __name__ == '__main__':
    # X, Y = generate_linear()
    # nn1 = NN(layers=[2, 3, 3, 1], activations=['sigmoid', 'sigmoid', 'sigmoid'], lr=0.01)
    # nn1.train(X, Y, batch_size=1, epochs=1500, lr=0.01)
    # nn1.test(X, Y)

    x, y = generate_XOR_easy()

    nn2 = NN(layers=[2, 3, 3, 1], activations=['relu', 'relu', 'relu'], lr=0.01)
    nn2.train(x, y, batch_size=2, epochs=90000, lr=0.01)
    nn2.test(x, y)


    #### testing ####
    # loss = 0.0
    # num_correct = 0
    # for i in range(len(X)):
    #     # import pdb
    #     # pdb.set_trace()
    #     tmp = X[i:i+1].T 
    #     z_s, a_s = nn.forward(tmp)

    #     prediction = a_s[-1][0][0]
    #     loss += np.linalg.norm(a_s[-1] - Y[i])

    #     predict_label = 0
    #     if prediction >= 0.5 : 
    #         predict_label = 1
    #     if predict_label == Y[i] :
    #         num_correct += 1

    #     print(f"Iter {i} |  Ground truth: {Y[i]} |  Prediction: {prediction}.")
    
    # print(f"loss={loss / len(X)} accuracy={100 * num_correct / len(X)}%")


        
        

        
