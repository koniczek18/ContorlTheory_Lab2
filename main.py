import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def printer(A1, A2, B1, B2, C1, C2, D1, D2):
    print('A1', A1)
    print('A2', A2)
    print('B1', B1)
    print('B2', B2)
    print('C1', C1)
    print('C2', C2)
    print('D1', D1)
    print('D2', D2)
    '''
    funkcja do wyświetlania 2 systemów celem ich porównania
    '''


def zadanie1(active):
    if active:
        D = np.array([0])
        # system 1
        A1 = np.array([-2])
        B1 = np.array([10])
        C1 = np.array([1])
        ss1 = sig.StateSpace(A1, B1, C1, D)
        tf1 = sig.TransferFunction(np.array([10]), np.array([1, 2]))
        # system 2
        A2 = np.array([[0, 1], [-0.5, 0]])
        B2 = np.array([[0], [2]])
        C2 = np.array([1, 0])
        ss2 = sig.StateSpace(A2, B2, C2, D)
        tf2 = sig.TransferFunction(np.array([4]), np.array([2, 0, 1]))
        # system3
        A3 = np.array([[0, 1, 0], [0, 0, 1], [-12, -16, -7]])
        B3 = np.array([[0], [-2], [20]])
        C3 = np.array([1, 0, 0])
        ss3 = sig.StateSpace(A3, B3, C3, D)
        tf3 = sig.TransferFunction(np.array([-2, 6]), np.array([1, 7, 16, 12]))
        # przekształcenia
        Atf1, Btf1, Ctf1, Dtf1 = sig.tf2ss(np.array([10]), np.array([1, 2]))
        Atf2, Btf2, Ctf2, Dtf2 = sig.tf2ss(np.array([4]), np.array([2, 0, 1]))
        Atf3, Btf3, Ctf3, Dtf3 = sig.tf2ss(np.array([-2, 6]), np.array([1, 7, 16, 12]))
        # porównianie
        printer(A1, Atf1, B1, Btf1, C1, Ctf1, D, Dtf1)
        printer(A2, Atf2, B2, Btf2, C2, Ctf2, D, Dtf2)
        printer(A3, Atf3, B3, Btf3, C3, Ctf3, D, Dtf3)
        # 1.4 Macierze A,B,C,D otrzymane przez funkcje scipy.signal.StateSpace oraz scipy.signal.tf2ss są różne
        #    Jest to spowodowane tym że ta sama transmitancja może mieć różną postać w równaniach stanu
        #
        # symulacje
        Ttf1, Ytf1 = sig.step2(tf1)
        Tss1, Yss1 = sig.step2(ss1)
        Ttran1, Ytran1 = sig.step2([Atf1, Btf1, Ctf1, Dtf1])
        Ttf2, Ytf2 = sig.step2(tf2)
        Tss2, Yss2 = sig.step2(ss2)
        Ttran2, Ytran2 = sig.step2([Atf2, Btf2, Ctf2, Dtf2])
        Ttf3, Ytf3 = sig.step2(tf3)
        Tss3, Yss3 = sig.step2(ss3)
        Ttran3, Ytran3 = sig.step2([Atf3, Btf3, Ctf3, Dtf3])
        # System 1
        plt.figure(0)
        plt.title('Tf1')
        plt.plot(Ttf1, Ytf1, color='b')
        plt.figure(1)
        plt.title('Ss1')
        plt.plot(Tss1, Yss1, color='r')
        plt.figure(2)
        plt.title('Transformed 1')
        plt.plot(Ttran1, Ytran1, color='g')
        # System 2
        plt.figure(3)
        plt.title('Tf2')
        plt.plot(Ttf2, Ytf2, color='b')
        plt.figure(4)
        plt.title('Ss2')
        plt.plot(Tss2, Yss2, color='r')
        plt.figure(5)
        plt.title('Transformed 1')
        plt.plot(Ttran2, Ytran2, color='g')
        # System 3
        plt.figure(6)
        plt.title('Tf3')
        plt.plot(Ttf3, Ytf3, color='b')
        plt.figure(7)
        plt.title('Ss3')
        plt.plot(Tss3, Yss3, color='r')
        plt.figure(8)
        plt.title('Transformed 3')
        plt.plot(Ttran3, Ytran3, color='g')
        # 1.5 Przebieg wyjścia jest taki sam dla wszystkich (StateSpace, TF, TF2SS) reprezentacji
        #    Trajektorie zmiennych stanów nie muszą być takie same
        # TODO 1.6
        Tnztf1, Ynztf1 = sig.step(tf1, np.array([3]))
        Tnzss1, Ynzss1 = sig.step(ss1, np.array([3]))
        Tnztran1, Ynztran1 = sig.step([Atf1, Btf1, Ctf1, Dtf1], np.array([3]))
        # TODO 2 system
        # TODO 3 system
        plt.figure(9)
        plt.title('Tf1 - ini [3]')
        plt.plot(Tnztf1, Ynztf1, color='b')
        plt.figure(10)
        plt.title('Ss1 - ini [3]')
        plt.plot(Tnzss1, Ynzss1, color='r')
        plt.figure(11)
        plt.title('Transformed 1 - ini [3]')
        plt.plot(Tnztran1, Ynztran1, color='g')
        plt.plot()


if __name__ == '__main__':
    zadanie1(True)