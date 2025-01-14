import numpy as np
import cmath as cm

def p_func(x, v, tau, ln_K, r, sigma, rho, a, b, u, num_iter):
    # Take the real part of the integration result
    return 0.5 + integrale(x, v, tau, ln_K, r, sigma, rho, a, b, u, num_iter).real / cm.pi

def f_func(x, v, tau, ln_K, r, sigma, rho, a, b, u, phi):
    d = cm.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi + d) / (b - rho * sigma * phi - d)
    D = ((b - rho * sigma * phi + d) / sigma**2) * ((1 - cm.exp(d * tau)) / (1 - g * cm.exp(d * tau)))
    C = (
        r * phi * tau * 1j
        + (a / sigma**2)
        * (
            (b - rho * sigma * phi + d) * tau
            - 2 * cm.log((1 - g * cm.exp(d * tau)) / (1 - g))
        )
    )
    # Take the real part of the result
    return (
        cm.exp(-1j * phi * ln_K)
        * cm.exp(C + D * v + 1j * phi * x)
        / (1j * phi * (1 + phi)**2)
    ).real  # Extract only the real part

def integrale(x, v, tau, ln_K, r, sigma, rho, a, b, u, num_iter):
    y = np.random.uniform(1e-6, 1-1e-6, num_iter)
    phi = 1 / y - 1
    f_val = [f_func(x, v, tau, ln_K, r, sigma, rho, a, b, u, phi_i) for phi_i in phi]
    # Compute the mean (real part already extracted in f_func)
    return np.mean(f_val)

def call_price(stock_price, stock_vol, strike, maturity, interest_rate, kapa, beta, sigma, rho, t, num_iter=1000):
    # Parameters for the Heston model
    a_1, a_2 = 0.5, kapa * beta
    b_1, b_2 = kapa - rho * sigma, kapa
    u_1, u_2 = 0.5, -0.5  # Remaining u_1 and u_2
    tau = maturity - t

    # Compute probabilities
    p_1 = p_func(np.log(stock_price), stock_vol, tau, np.log(strike), interest_rate, sigma, rho, a_1, b_1, u_1, num_iter)
    p_2 = p_func(np.log(stock_price), stock_vol, tau, np.log(strike), interest_rate, sigma, rho, a_2, b_2, u_2, num_iter)

    # Compute call price
    return stock_price * p_1 - strike * np.exp(-interest_rate * tau) * p_2
    