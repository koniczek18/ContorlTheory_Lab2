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
    print('\n')
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
        # 1.3 Macierze A,B,C,D otrzymane przez funkcje scipy.signal.StateSpace oraz scipy.signal.tf2ss są różne
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
        # wyświetlanie wykresów zmienić zmieniając instukjcę warunkową przed kodem
        #
        # System 1
        if True:
            plt.figure('System 1')
            plt.title('System 1')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(Ttf1, Ytf1, color='b',label='Tf1',linestyle='dashed')
            plt.plot(Tss1, Yss1, color='r',label='Ss1',linestyle='dashdot')
            plt.plot(Ttran1, Ytran1, color='g',label='Transformed 1',linestyle='dotted')
            plt.legend()
        # System 2
        if True:
            plt.figure('System 2')
            plt.title('System 2')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(Ttf2, Ytf2, color='b', label='Tf2',linestyle='dashed')
            plt.plot(Tss2, Yss2, color='r', label='Ss2',linestyle='dashdot')
            plt.plot(Ttran2, Ytran2, color='g', label='Transformed 2',linestyle='dotted')
            plt.legend()
        # System 3
        if True:
            plt.figure('System 3')
            plt.title('System 3')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.plot(Ttf3, Ytf3, color='b', label='Tf3',linestyle='dashed')
            plt.plot(Tss3, Yss3, color='r', label='Ss3',linestyle='dashdot')
            plt.plot(Ttran3, Ytran3, color='g', label='Transformed 3',linestyle='dotted')
            plt.legend()
        # 1.4a Przebieg wyjścia jest taki sam dla wszystkich (StateSpace, TF, TF2SS) reprezentacji
        # 1.4b Trajektorie zmiennych stanów nie muszą być takie same
        #System 1
        ssSys1=sig.lti(A1, B1, C1, D)
        TranSys1=sig.lti(Atf1, Btf1, Ctf1, Dtf1)
        Tnzss1, Ynzss1 = sig.step2(ssSys1, X0=np.array([3]))
        Tnztran1, Ynztran1 = sig.step2(TranSys1, X0=np.array([3]))
        # System 2
        ssSys2 = sig.lti(A2, B2, C2, D)
        TranSys2 = sig.lti(Atf2, Btf2, Ctf2, Dtf2)
        Tnzss2, Ynzss2 = sig.step2(ssSys2, X0=np.array([3,3]))
        Tnztran2, Ynztran2 = sig.step2(TranSys2, X0=np.array([3,3]))
        # System 3
        ssSys3 = sig.lti(A3, B3, C3, D)
        TranSys3 = sig.lti(Atf3, Btf3, Ctf3, Dtf3)
        Tnzss3, Ynzss3 = sig.step2(ssSys3, X0=np.array([3,3,3]))
        Tnztran3, Ynztran3 = sig.step2(TranSys3, X0=np.array([3,3,3]))
        #System 1
        if True:
            plt.figure('System 1 - initial conditions')
            plt.title('System 1 - initial conditions')
            plt.plot(Tnzss1, Ynzss1, color='b',label='Ss1 - ini [3]',linestyle='dashed')
            plt.plot(Tnztran1, Ynztran1, color='r',label='Transformed1 - ini [3]',linestyle='dotted')
            plt.legend()
        if True:
            plt.figure('System 2 - initial conditions')
            plt.title('System 2 - initial conditions')
            plt.plot(Tnzss2, Ynzss2, color='b', label='Ss2 - ini [3]', linestyle='dashed')
            plt.plot(Tnztran2, Ynztran2, color='r', label='Transformed2 - ini [3]', linestyle='dotted')
            plt.legend()
        if True:
            plt.figure('System 3 - initial conditions')
            plt.title('System 3 - initial conditions')
            plt.plot(Tnzss3, Ynzss3, color='b', label='Ss3 - ini [3]', linestyle='dashed')
            plt.plot(Tnztran3, Ynztran3, color='r', label='Transformed3 - ini [3]', linestyle='dotted')
            plt.legend()
        plt.show()
        # 1.5a Dla różnych reprezentacji dobór warunków początkowych nie jest równoznaczny
        #      i nie uzyskuje się tych samych przebiegów
        # 1.5b Definicja transformaty Laplace'a, zakłada zerowe warunki początkowe, więc aby przestawić układ z niezerowymi war. pocz.
        #      musimy wprowadzić zmiany w definicji, a w rezultacie w transmitancji

def zadanie2(active):
    if active:
        #System 1
        A1=np.array([[-4,-1],[-2,-1]])
        B1=np.array([[2],[1]])
        C1=np.array([3,-4])
        D1=np.array([1])
        tf1=sig.ss2tf(A1,B1,C1,D1)
        print('System 1\nnum',tf1[0],'\nden',tf1[1],'\n')
        # System 2
        A2 = np.array([[-1,0,1], [-6,-3,5],[-5,-2,4]])
        B2 = np.array([[0], [1],[1]])
        C2 = np.array([1,1,1])
        D2 = np.array([0])
        tf2 = sig.ss2tf(A2, B2, C2, D2)
        print('System 2\nnum', tf2[0], '\nden', tf2[1], '\n')
        # System 3
        A3 = np.array([[-3, 1.25,-0.75, -2.75], [6, 3,-3.5,-6], [0,-1,0,1],[-6, 5,-4.5,-6]])
        B3 = np.array([[0.5], [1], [0],[1]])
        C3 = np.array([2,0,0,0])
        D3 = np.array([0])
        tf3 = sig.ss2tf(A3, B3, C3, D3)
        print('System 3\nnum', tf3[0], '\nden', tf3[1], '\n')
        # wyniki w niektórych przypadkach zawierają błędy spowodowane błędami numerycznymi
        # 2.2a - Nie, równania te opisują konkretny system, a 1 system może mieć 1 reprezentację w postaci transmitancji,
        #        ale wiele w postaci równań stanu
        # 2.2b - Zależność G(s)=C*inv(s*I-A)*B+D obowiązuje w każdym przypadku

if __name__ == '__main__':
    zadanie1(True)
    zadanie2(True)