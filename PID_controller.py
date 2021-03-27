
import numpy as np
from numpy.core.function_base import linspace
from scipy import signal
import matplotlib.pyplot as plt

class PID_using_transfer_function:
    def __init__(self, plant: np.ndarray, pidCoeffs:np.ndarray, t) -> None:
        self.plant = plant
        self.pidCoeff = pidCoeffs
        self.T = t

    def PID_controller(self):
        kp, ki, kd = self.pidCoeff
        tfNum, tfDen = self.plant
        nPID = [kd, kp, ki]
        dPID = [1, 0]
        num = np.convolve(nPID, tfNum)
        den = np.convolve(dPID, tfDen)
        return signal.TransferFunction(num,den + num)

    def plot_the_figuers(self):
        sysPID = self.PID_controller()
        plant = signal.TransferFunction(self.plant[0], self.plant[1])
        t = np.linspace(0,self.T)
        _, yPID = signal.step(sysPID, T=t)
        _, yPlant = signal.step(plant, T=t)
        y3 = np.ones(t.shape[0])

        plt.figure(1)
        plt.plot(t,yPID,'k-')
        plt.plot(t,yPlant,'b-')
        plt.plot(t,y3, 'r--')
        plt.legend(['PID step response', 'Plant response without PID'],loc='best')
        plt.xlabel('Time')
        plt.show()

#system1
# plant1 = [[0.66],[6.7, 1]]
# controller1 = [20, 2, 0]
# sys1 = PID_using_transfer_function(plant1, controller1, 20)
# sys1.plot_the_figuers()

#system2
num = np.convolve(np.array([0.87]), np.array([11.6, 1]))
den = np.convolve(np.array([3.89, 1]), np.array([18.8, 1]))
plant2 = [ num, den]
controller2 = [20, 2, 0]
sys2 = PID_using_transfer_function(plant2, controller2, 20)
sys2.plot_the_figuers()
        

