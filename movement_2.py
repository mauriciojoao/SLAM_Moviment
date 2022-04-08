import math
import numpy as np
import matplotlib.pyplot as plt

#−−−−−−−− Drawing Vehicle −−−−−#
def DrawRobot(Xr):
    plt.grid(b = True)
    plt.axis([-20, 20, -5, 30])
    plt.title('Uncertainty bounds for Ackermann model')
    plt.xlabel('x')
    plt.ylabel('y')
    p = 0.02
    xmin, xmax, ymin, ymax = plt.axis()
    l1 = (xmax - xmin) * p
    l2 = (ymax - ymin) * p
    P = np.array([[-1, 1, 0, -1], [-1, -1, 3, -1]])
    theta = Xr[2] - np.pi/2
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype = float)
    P = rotation.dot(P)
    P[0, :] = P[0, :] * l1 + Xr[0]
    P[1, :] = P[1, :] * l2 + Xr[1]
    plt.plot(P[0, :], P[1, :], color = 'blue')
    plt.plot(Xr[0], Xr[1], 'b+')

#Vetor de estado "X" com as informacoes iniciais:
x = 0
y = 0
theta = 0
X = np.array([x,y,theta])
# X = [x,y,theta]
#X = np.array[
#    [],
#    [],
#    []
#]

print(X)

#função com as velocidades:
def vel():
    vel_s = 0.5
    vel_theta = 0.0
    w = np.array([vel_s,vel_theta])
    #w = np.array[
    #    [],
    #    []
    #]
    
    # w = [vel_s,vel_theta]
    return w

#função que retorna o vetor q:
def movement(X,vel):
    dT = 1
    q = np.array([math.cos(X[2])*vel[0], math.sin(X[2])*vel[0], vel[1]])*dT
    #q = [math.cos(theta)*0.5, math.sin(theta)*0.5, 0.1]
    return q 




#−−−−−−−− Set up graphics −−−−−−−−−−−#


#for para simular a execução:
for i in range(0, 100):
    X = X + movement(X,vel())
    print(X)
    

    if (i % 1) == 0: # Comentar para animar
        plt.clf()

        DrawRobot(X)
        plt.pause(0.1)
    if (i % 20) == 0:
        X[2] += np.pi/2
plt.show()

#print(X)