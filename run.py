import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,3,4,7,9,10,11,18])
y = np.array([5,12,12,20,32,30,33,60])

def Gradient_descent(x):
    alpha = 0.001; m = len(x)
    theta0,theta1 = np.random.randint(1,6,2) #initialize two parameters with int between [1,6)
    while(True):
        h = theta0 + theta1*x
        J = 1/(2*m)*np.sum(((h-y)**2))
        theta0 = theta0 - alpha*1/m*np.sum((h-y))
        theta1 = theta1 - alpha*1/m*np.sum(((h-y)*x))
        print(J,theta0,theta1)
        if J <=5:
            return theta0, theta1
            break

if __name__ == "__main__":
    theta0,theta1 = Gradient_descent(x)
    
    x1 = np.linspace(np.min(x),np.max(x)+1)
    y1 = theta0 + theta1*x1
    plt.scatter(x,y,marker='x')
    plt.plot(x1,y1)
    plt.show()
