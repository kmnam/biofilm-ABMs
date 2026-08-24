"""
Authors:
    Kee-Myoung Nam

Last updated:
    8/24/2026
"""

import sys
import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import quad
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

########################################################################
def simulate(n, alpha, beta, lambdaA, lambdaB, K, gamma0, sigma, rng):
    """
    Simulate the asymmetry index calculation for the given number of 
    sister-cell pairs. 

    Parameters
    ----------
    n : int
        Number of sister-cell pairs. 
    alpha : float
        Signal value in phenotype A (initial phenotype).  
    beta : float 
        Signal value in phenotype B. 
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    lambdaB : float
        Rate parameter for lifetime distribution of phenotype B. 
    K : float
        Doubling-time coefficient. 
    gamma0 : float
        Mean growth rate. 
    sigma : float
        Standard deviation of growth rate. 
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Sampled (non-negative) asymmetry index values. 
    """
    # Sample growth rates from a common normal distribution 
    g1 = rng.normal(loc=gamma0, scale=sigma, size=n)
    g2 = rng.normal(loc=gamma0, scale=sigma, size=n)

    q = []
    for i in range(n):
        # Get the corresponding growth times 
        div_t1 = K / g1[i]
        div_t2 = K / g2[i]

        # Sample a sequence of state lifetimes for each cell 
        #
        # If the cell is in state A (an even number of lifetimes have been
        # sampled), then sample from an exponential distribution with rate
        # lambdaA
        #
        # Otherwise, sample from an exponential distribution with rate lambdaB
        seq_t1 = []
        seq_t2 = []
        while np.sum(seq_t1) < div_t1:
            scale = 1 / lambdaA if len(seq_t1) % 2 == 0 else 1 / lambdaB
            seq_t1.append(rng.exponential(scale=scale))
        while np.sum(seq_t2) < div_t2:
            scale = 1 / lambdaA if len(seq_t2) % 2 == 0 else 1 / lambdaB
            seq_t2.append(rng.exponential(scale=scale))

        # Get the number of switching events up to the division time 
        n_switch1 = len(seq_t1) - 1
        n_switch2 = len(seq_t2) - 1

        # If there were no switches up to the division time, the mean signal
        # is simply alpha 
        if n_switch1 == 0:
            total_t1 = div_t1
            signal1 = alpha
        # If there were an even number of switches up to the division time, 
        # then the cell was in state A at the division time 
        elif n_switch1 % 2 == 0:
            total_t1 = div_t1 - np.sum(seq_t1[1:-1:2])
            signal1 = (alpha * total_t1 + beta * (div_t1 - total_t1)) / div_t1
        # If there were an odd number of switches up to the division time, 
        # then the cell was in state B at the division time 
        else:
            total_t1 = np.sum(seq_t1[:-1:2])
            signal1 = (alpha * total_t1 + beta * (div_t1 - total_t1)) / div_t1

        # Similarly calculate the mean signal in cell 2
        if n_switch2 == 0:
            total_t2 = div_t2
            signal2 = alpha
        elif n_switch2 % 2 == 0:
            total_t2 = div_t2 - np.sum(seq_t2[1:-1:2])
            signal2 = (alpha * total_t2 + beta * (div_t2 - total_t2)) / div_t2
        else:
            total_t2 = np.sum(seq_t2[:-1:2])
            signal2 = (alpha * total_t2 + beta * (div_t2 - total_t2)) / div_t2

        # Calculate the log-ratio between the two signals 
        qi = np.abs(np.log(signal1) - np.log(signal2))
        q.append(qi)

    return np.array(q)

########################################################################
def chi(y, t, lambdaA, lambdaB):
    """
    Calculate the function \chi_{M_i}.

    Parameters
    ----------
    y : float 
        Input value; represents a normalized fraction of time spent by a 
        cell in phenotype A from birth to division. 
    t : float
        Fixed division time of the cell. 
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    lambdaB : float
        Rate parameter for lifetime distribution of phenotype B.

    Returns
    -------
    Output value. 
    """
    kA = lambdaA * y
    kB = lambdaB * (1 - y)
    xi = kA * kB
    arg = 2 * t * np.sqrt(xi)

    # Evaluate I0() and I1() using the exponentially scaled versions, and 
    # multiply by a factor of exp(arg)
    term1 = t * np.exp(-t * (kA + kB) + arg)
    term2 = lambdaA * i0e(arg)
    term3 = np.sqrt(lambdaA * lambdaB * y / (1 - y)) * i1e(arg)

    return term1 * (term2 + term3)

########################################################################
def psi(y, lambdaA, lambdaB, K, gamma0, sigma):
    """
    Calculate the function \psi.

    Parameters
    ----------
    y : float 
        Input value; represents a normalized fraction of time spent by a 
        cell in phenotype A from birth to division. 
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    lambdaB : float
        Rate parameter for lifetime distribution of phenotype B.
    K : float
        Doubling-time coefficient. 
    gamma0 : float
        Mean growth rate. 
    sigma : float
        Standard deviation of growth rate. 

    Returns
    -------
    Output value. 
    """
    integrand = lambda t: (
        K / (t * t) * chi(y, t, lambdaA, lambdaB) * 
        norm.pdf(K / t, loc=gamma0, scale=sigma)
    )
    return quad(integrand, 0, np.inf)[0]

########################################################################
def omega(lambdaA, K, gamma0, sigma):
    """
    Calculate the constant omega. 

    Parameters
    ----------
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    K : float
        Doubling-time coefficient. 
    gamma0 : float
        Mean growth rate. 
    sigma : float
        Standard deviation of growth rate. 

    Returns
    -------
    Output value. 
    """
    integrand = lambda t: (
        np.exp(-lambdaA * t) * (K / (t * t)) *
        norm.pdf(K / t, loc=gamma0, scale=sigma)
    )
    return quad(integrand, 0, np.inf)[0]

########################################################################
def density_Q_nonzero(q, alpha, beta, lambdaA_1, lambdaB_1, lambdaA_2, lambdaB_2,
                      K, gamma0_1, sigma1, gamma0_2, sigma2):
    """
    Calculate the nonzero component of the density of the (signed) asymmetry
    index Q. 

    Here, q is assumed to be an array of nonzero input values.

    Parameters
    ----------
    q : `numpy.ndarray`
        Array of input values. 
    alpha : float
        Signal value in phenotype A (initial phenotype).  
    beta : float 
        Signal value in phenotype B. 
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    lambdaB : float
        Rate parameter for lifetime distribution of phenotype B. 
    K : float
        Doubling-time coefficient. 
    gamma0 : float
        Mean growth rate. 
    sigma : float
        Standard deviation of growth rate.

    Returns
    -------
    Nonzero component of the density of Q. 
    """
    rho = alpha - beta

    # Determine the interval S
    if rho == 0:
        raise RuntimeError()
    elif rho > 0:
        Smin = np.log(beta)
        Smax = np.log(alpha)
    else:
        Smin = np.log(alpha)
        Smax = np.log(beta)

    # Determine the interval V for each value in q
    Vmin = np.maximum((beta * np.exp(-q) - beta) / rho, 0)
    Vmax = np.minimum((alpha * np.exp(-q) - beta) / rho, 1)

    # Identify the values of q for which log(alpha) - q lies in S
    ind1 = ((np.log(alpha) - q > Smin) & (np.log(alpha) - q < Smax))

    # Identify the values of q for which q + log(alpha) lies in S
    ind2 = ((np.log(alpha) + q > Smin) & (np.log(alpha) + q < Smax))

    # Calculate the first and second terms (atomic-continuous and
    # continuous-atomic)
    #
    # Weight 1 corresponds to cell 1, weight 2 to cell 2
    weight1 = alpha * omega(lambdaA_1, K, gamma0_1, sigma1) / np.abs(rho)
    weight2 = alpha * omega(lambdaA_2, K, gamma0_2, sigma2) / np.abs(rho)
    term1 = weight1 * np.exp(q) * ind2
    term2 = weight2 * np.exp(-q) * ind1

    # For each value of q ... 
    term3 = np.zeros(q.size)
    for i in range(q.size):
        # Multiply the omega_1 value by a corresponding psi_2 value 
        if term1[i] != 0:
            arg = (alpha * np.exp(q[i]) - beta) / rho
            term1[i] *= psi(arg, lambdaA_2, lambdaB_2, K, gamma0_2, sigma2)

        # Multiply the omega_2 value by a corresponding psi_1 value
        if term2[i] != 0:
            arg = (alpha * np.exp(-q[i]) - beta) / rho
            term2[i] *= psi(arg, lambdaA_1, lambdaB_1, K, gamma0_1, sigma1)

        # Calculate the third term, as long as the integration interval is 
        # non-empty 
        if Vmin[i] < Vmax[i]:
            integrand = lambda m: (
                (np.exp(q[i]) * (beta + rho * m) / np.abs(rho)) * 
                psi(
                    (np.exp(q[i]) * (beta + rho * m) - beta) / rho,
                    lambdaA_2, lambdaB_2, K, gamma0_2, sigma2
                ) *\
                psi(m, lambdaA_1, lambdaB_1, K, gamma0_1, sigma1)
            )
            term3[i] = quad(integrand, Vmin[i], Vmax[i])[0]

    return term1 + term2 + term3

########################################################################
def density_Q_zero_weight(lambdaA_1, lambdaA_2, K, gamma0_1, sigma1, gamma0_2,
                          sigma2):
    """
    Calculate the weight of the atomic component of the density of the
    (signed) asymmetry index Q at zero. 

    Parameters
    ----------
    lambdaA : float
        Rate parameter for lifetime distribution of phenotype A. 
    K : float
        Doubling-time coefficient. 
    gamma0 : float
        Mean growth rate. 
    sigma : float
        Standard deviation of growth rate.

    Returns
    -------
    Weight of the atomic component at zero. 
    """
    z1 = omega(lambdaA_1, K, gamma0_1, sigma1)
    z2 = omega(lambdaA_2, K, gamma0_2, sigma2)
    return z1 * z2

########################################################################
if __name__ == '__main__':
    # Define model parameters 
    alpha = 0.35
    beta = 0.15
    L0 = 1.0
    R = 0.8
    K = np.log((6 * L0 + 10 * R) / (3 * L0 + 4 * R))
    gamma0_1 = 1.12
    sigma1 = 0.2 * gamma0_1

    # Get the state lifetimes from input arguments 
    tauA_1 = float(sys.argv[1])
    tauB_1 = float(sys.argv[2])
    tauA_2 = float(sys.argv[3])
    tauB_2 = float(sys.argv[4])
    lambdaA_1 = 1.0 / tauA_1
    lambdaB_1 = 1.0 / tauB_1
    lambdaA_2 = 1.0 / tauA_2
    lambdaB_2 = 1.0 / tauB_2

    # Fix the fold difference in mean growth rate between cell 1 and cell 2
    gamma_fold = float(sys.argv[5])
    gamma0_2 = gamma_fold * gamma0_1
    sigma2 = 0.2 * gamma0_2

    # Initialize random number generator 
    rng = np.random.default_rng(1234567890)

    # Calculate the theoretical densities for nonzero q
    qmax = np.abs(np.log(alpha) - np.log(beta))
    q0 = np.linspace(-0.9999 * qmax, 0.9999 * qmax, 500)
    pQ_nz1 = density_Q_nonzero(
        q0, alpha, beta, lambdaA_1, lambdaB_1, lambdaA_2, lambdaB_2, K,
        gamma0_1, sigma1, gamma0_2, sigma2
    )
    pQ_nz2 = density_Q_nonzero(
        q0, beta, alpha, lambdaB_1, lambdaA_1, lambdaB_2, lambdaA_2, K,
        gamma0_1, sigma1, gamma0_2, sigma2
    )

    # Calculate the zero weights 
    zero_w1 = density_Q_zero_weight(
        lambdaA_1, lambdaA_2, K, gamma0_1, sigma1, gamma0_2, sigma2
    )
    zero_w2 = density_Q_zero_weight(
        lambdaB_1, lambdaB_2, K, gamma0_1, sigma1, gamma0_2, sigma2
    )

    # Parse the input data and fit the Gaussian kernel width 
    data = np.loadtxt('logratios.txt') * np.log(2)
    data_near_zero = data[data < 0.15]
    signs = rng.choice([-1, 1], size=data_near_zero.shape, replace=True)
    data_near_zero *= signs
    _, eps = norm.fit(data_near_zero)
    print('Fitted Gaussian kernel width:', eps)
    
    # Enlarge the input mesh slightly, so that it ranges to 1.1 * qmax with
    # the same increment 
    dq = q0[1] - q0[0]
    n_pad = int(np.ceil((1.1 * qmax - q0[-1]) / dq))
    q_ext = np.zeros(q0.size + 2 * n_pad, dtype=np.float64)
    q_ext[n_pad:-n_pad] = q0
    for i in range(n_pad - 1, -1, -1):
        q_ext[i] = q_ext[i + 1] - dq
        q_ext[-(i + 1)] = q_ext[-(i + 2)] + dq
    assert np.all(np.abs(q_ext[1:] - q_ext[:-1] - dq) < 1e-8)

    # Fill in the nonzero portions of the theoretical densities  
    pQ_nz1_ext = np.zeros(q_ext.size, dtype=np.float64)
    pQ_nz2_ext = np.zeros(q_ext.size, dtype=np.float64)
    pQ_nz1_ext[n_pad:-n_pad] = pQ_nz1
    pQ_nz2_ext[n_pad:-n_pad] = pQ_nz2
    assert np.all(pQ_nz1_ext[:n_pad] == 0)
    assert np.all(pQ_nz1_ext[-n_pad:] == 0)
    assert np.all(pQ_nz2_ext[:n_pad] == 0)
    assert np.all(pQ_nz2_ext[-n_pad:] == 0)

    # Convolve both densities with a Gaussian kernel
    pQ_nz1_convolved = gaussian_filter1d(
        pQ_nz1_ext, sigma=(eps / dq), mode='constant', cval=0.0
    )
    pQ_nz2_convolved = gaussian_filter1d(
        pQ_nz2_ext, sigma=(eps / dq), mode='constant', cval=0.0
    )

    # Add in a Gaussian peak to account for q = 0 
    pQ_nz1_convolved += zero_w1 * norm.pdf(q_ext, loc=0, scale=eps)
    pQ_nz2_convolved += zero_w2 * norm.pdf(q_ext, loc=0, scale=eps)

    # Write the values to file and plot the distribution
    header = (
        'tauA_1 = {:.10f}, tauB_1 = {:.10f}, tauA_2 = {:.10f}, '
        'tauB_2 = {:.10f}, gamma_fold = {:.10f}'.format(
            tauA_1, tauB_1, tauA_2, tauB_2, gamma_fold
        )
    )
    np.savetxt(
        sys.argv[6],
        np.vstack((
            np.hstack((
                q_ext.reshape(-1, 1), pQ_nz1_convolved.reshape(-1, 1)
            )),
            np.hstack((
                q_ext.reshape(-1, 1), pQ_nz2_convolved.reshape(-1, 1)
            ))
        )),
        header=header,
        comments='# ',
        delimiter='\t'
    )

