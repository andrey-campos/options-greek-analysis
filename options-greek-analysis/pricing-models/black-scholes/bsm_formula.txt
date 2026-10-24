# -*- coding: utf 8 -*-

import numpy as np 
from scipy.stats import norm 

# function to calculate d1 and d2 based off the numerous variables
def calc_d1_d2(S, X, t, r, q, sigma) -> tuple[float, float]:
    """
    ### args:
    - S: Underlying stock price
    - X: Exercise Price 
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate
    - sigma: volatility 

    ### returns:
    d1 and d2, values used in CDF as an intermediate calculation
    """

    d1: float = (np.log(S/X) + (r - q + (sigma ** 2) / 2) * (t)) / (sigma * np.sqrt(t))
    d2: float = d1 - (sigma * np.sqrt(t))

    return d1, d2

# function to calculate call option values per black-scholes-merton formula
def bsm_call(S, X, t, r, q, d1, d2) -> float:
    """
    ### args: 
    - S: Underlying stock price
    - X: Exercise Price 
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate

    ### returns:
        the value of a european call option
    """

    # 4) calculate N(d1) and N(d2) as intermediate calculations
    N_d1: float = norm.cdf(d1)
    N_d2: float = norm.cdf(d2)

    return ((S * np.exp(-q*t)) * N_d1) - ((X * np.exp(-r*t)) * N_d2)
    
# function to calculate put option values per black-scholes-merton formula
def bsm_put(S, X, t, r, q, d1, d2) -> float:
    """
    ### args: 
    - S: Underlying stock price
    - X: Exercise Price 
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate

    ### returns:
        the value of a european put option
    """

    # 6) calculate N(-d1) and N(-d2) as intermediate calculations
    N_d1: float = norm.cdf(-d1)
    N_d2: float = norm.cdf(-d2)

    return ((X * np.exp(-r*t)) * N_d2) - ((S * np.exp(-q*t)) * N_d1)

# function to check our calculations using Put-Call parity
def put_call_parity(S, X, t, r, q) -> float:
    """
    ### args:
    - S: Underlying stock price
    - X: Exercise Price 
    - t: time to expiration (also known as T-t)
    - r: interest rate
    - q: yield rate

    ## returns:
        Call - Put 
    """

    return S * np.exp(-q*t) - X * np.exp(-r*t)


if __name__ == "__main__":
    invalid_input = True
    
    # 1) Take in inputs from user for intermediate calculations and validate user input
    while invalid_input:
        stock = float(input("Enter underlying stock price (S): ")) 
        exercise = float(input("Enter exercise price (X): ")) 
        time_to_ex = float(input("Enter time to expiration (T-t) or (t): ")) 
        volatility = float(input("Enter volatility (sigma): ")) 
        interest_rate= float(input("Enter interest rate (r): ")) 

        # validate all inputs to be decimals greater than zero
        inputs = [stock, exercise, time_to_ex, volatility, interest_rate]
        if all(input > 0 for input in inputs):
            invalid_input = False
        else:
            print("All inputs until this point must be greater than zero.")

    # take in yield rate separately 
    yield_rate = float(input("Enter yield rate (q): ")) 

    # calculate and display d1 and d2 to ensure all calculations are correct
    d1, d2 = calc_d1_d2(
        S=stock,
        X=exercise,
        t=time_to_ex,
        r=interest_rate,
        q=yield_rate,
        sigma=volatility    
    )
    print(f"\nd1 = {d1:.4f}\nd2 = {d2:.4f}")

    # call both functions to get the test and see value of our european call and put options (for 2 months of expiration)
    call_value: float = bsm_call(
        S=stock,
        X=exercise,
        t=time_to_ex,
        r=interest_rate,
        q=yield_rate,
        d1=d1,
        d2=d2
    )

    print(f"\nThe value of our call option is (2 months to expiration): {call_value:.4f}")

    put_value: float = bsm_put(
        S=stock,
        X=exercise,
        t=time_to_ex,
        r=interest_rate,
        q=yield_rate,
        d1=d1,
        d2=d2
    )
    print(f"The value of our put option is (2 months to expiration): {put_value:.4f}")

    # 2) calculate the call value with the time of expiration now being 6 months -> also means you have to redefine d1, d2..
    d1_6m, d2_6m = calc_d1_d2(
        S=stock,
        X=exercise,
        t=0.5,
        r=interest_rate,
        q=yield_rate,
        sigma=volatility
    )

    call_value_6m: float = bsm_call(
        S=stock,
        X=exercise,
        t=0.5, # <- 6 months / 0.5 years
        r=interest_rate,
        q=yield_rate,
        d1=d1_6m,
        d2=d2_6m
    )
    print(f"\nThe value of our call option with extended time to expiration of 6 months: {call_value_6m:.4f}")

    # 3 ) Calculate put option value for 6 months to expiration
    put_value_6m: float = bsm_put(
        S=stock,
        X=exercise,
        t=0.5, # <- 6 months / 0.5 years
        r=interest_rate,
        q=yield_rate,
        d1=d1_6m,
        d2=d2_6m
    )
    print(f"The value of our put option with extended time to expiration of 6 months: {put_value_6m:.4f}")

    # 4) Using Put-Call parity check calculations for 6 months
    parity_6m: float = put_call_parity(
        S=stock,
        X=exercise,
        t=0.5,
        r=interest_rate,
        q=yield_rate,
    )

    # ensure both function and manual calculations yield same results..
    difference_6m = call_value_6m - put_value_6m
    print(f"\nPut-Call parity: {parity_6m:.4f} and the Actual difference (C-P) {difference_6m:.4f} (6 months expiration)")