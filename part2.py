import re
import numpy
import scipy
import scipy.optimize as optimize
import scipy.fft as fft
import matplotlib.pyplot as pl


def epsilon(arguments, *args):

    def epsilon_k(k, alpha, A, B, omega, L, a, ksi_0, ksi_1, mu_0, mu_1, g_k):

        def compute_lambda0(k, omega, ksi_0, mu_0):     # Calcule lambda0
            if k**2 > ksi_0*omega**2/mu_0:
                return numpy.sqrt(k**2-ksi_0*omega**2/mu_0)
            elif k**2 < ksi_0*omega**2/mu_0:
                return complex(0, 1)*numpy.sqrt(-(k**2-ksi_0*omega**2/mu_0))
            else:
                return 0

        def compute_lambda1(k, a, omega, ksi_1, mu_1):  # Calcule lambda1
            return numpy.sqrt(k**2 - ksi_1*omega**2/mu_1 + numpy.sqrt((k**2 - ksi_1*omega**2/mu_1)**2 + (a*omega/mu_1)**2))/numpy.sqrt(2) - numpy.sqrt(ksi_1*omega**2/mu_1 - k**2 + numpy.sqrt((k**2 - ksi_1*omega**2/mu_1)**2 + (a*omega/mu_1)**2))*complex(0, 1)/numpy.sqrt(2)

        # On les définit dès maintenant pour ne pas les recalculer à chaque fois
        lambda0 = compute_lambda0(k, omega, ksi_0, mu_0)
        lambda1 = compute_lambda1(k, a, omega, ksi_1, mu_1)

        def f(x):
            return (lambda0*mu_0-x)*numpy.exp(-lambda0*L) + (lambda0*mu_0+x)*numpy.exp(lambda0*L)

        def compute_chi(k, alpha):  # Calcule chi
            return g_k*((lambda0*mu_0-lambda1*mu_1)/f(lambda1*mu_1)-(lambda0*mu_0-alpha)/f(alpha))

        def compute_gamma(k, alpha):  # Calcule gamma
            return g_k*((lambda0*mu_0+lambda1*mu_1)/f(lambda1*mu_1)-(lambda0*mu_0+alpha)/f(alpha))

        # On les définit dès maintenant pour ne pas les recalculer à chaque fois
        chi = compute_chi(k, alpha)
        gamma = compute_gamma(k, alpha)

        if k**2 >= ksi_0*omega**2/mu_0:     # Formule du cours
            return (A+B*abs(k)**2)*(1/(2*lambda0)*(abs(chi)**2*(1-numpy.exp(-2*lambda0*L))+abs(gamma)**2*(numpy.exp(2*lambda0*L)-1)) + 2*L*numpy.real((chi*numpy.conjugate(gamma)))) + B*(lambda0/2)*(abs(chi)**2*(1-numpy.exp(-2*lambda0*L))+abs(gamma)**2*(numpy.exp(2*lambda0*L)-1)) - 2*B*lambda0**2*L*numpy.real((chi*numpy.conjugate(gamma)))
        else:
            return (A+B*abs(k)**2)*(L*(abs(chi)**2+abs(gamma)**2)+(complex(0, 1)/lambda0)*numpy.imag((chi*numpy.conjugate(gamma)*(1-numpy.exp(-2*lambda0*L))))) + B*L*abs(lambda0)**2*(abs(chi)**2+abs(gamma)**2) + complex(0, 1)*B*lambda0*numpy.imag((chi*numpy.conjugate(gamma)*(1-numpy.exp(-2*lambda0*L))))

    alpha = complex(arguments[0], arguments[1])     # Paramètre à minimiser

    omega = args[0]
    L = args[1]
    g = args[2]
    N = args[3]

    # # Définition des paramètres du matériau
    phi = 0.99  # porosity
    gamma_p = 7.0 / 5.0
    sigma = 14000.  # resitivity
    rho_0 = 1.2
    alpha_h = 1.02  # tortuosity
    c_0 = 340.0
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0**2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0**2)
    a = sigma * (phi**2) * gamma_p / ((c_0**2) * rho_0 * alpha_h)
    A, B = 1, 1

    # # On prend g comme dirac, donc sa transformée de Fourier est 1.
    g_k = 1

    S = epsilon_k(0, alpha, A, B, omega, L, a, ksi_0, ksi_1, mu_0, mu_1, g_k)

    for k in range(1, N):
        S += epsilon_k(k*numpy.pi/L, alpha, A, B, omega, L,
                       a, ksi_0, ksi_1, mu_0, mu_1, g_k)
        S += epsilon_k(-k*numpy.pi/L, alpha, A, B, omega,
                       L, a, ksi_0, ksi_1, mu_0, mu_1, g_k)

    return S


W = numpy.linspace(1, 30000, 200)
L = 0.01
def g(x): return 1 if x == 0 else 0


# On doit tronquer la somme des epsilon_k, pusique c'est une somme infinie. On définit ainsi un N maximuum.
N = 12

alpha_Re = []
alpha_Im = []

# On calcule alpha pour différents omega
for omega in W:
    args = omega, L, g, N       # Paramètres fixes
    initial_guess = [1, 1]      # Matrice de départ de l'optimisation
    result = optimize.minimize(epsilon, initial_guess, args)
    alpha_Re.append(result.x[0])
    alpha_Im.append(result.x[1])

# # plot alpha
pl.subplot(1, 2, 1)
pl.plot(W, alpha_Re)
pl.subplot(1, 2, 2)
pl.plot(W, alpha_Im)
pl.show()
pl.close()

print("End.")
