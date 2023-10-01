"""
Description: Compute the frequency spectrum
Usage: 
Input: each function has its own input data, please check
Output: each function has its own output data, please check
Modification by M.Sc. Rodney Eduardo Mora-Escalante
Contact to rodney.moraescalante@gmail.com
Creation day September 15,2021
Based on code by 
    Ph.D Mark Donelan (died in 2018)
    Emeritus Professor 
    Applied Marine Physics 
    University of Miami

"""


# Import Libraries

import numpy as np
# special libraries
import scipy.signal as signal
from scipy.fftpack import fft, fftshift
from numpy.linalg import norm


def _C_and_df(Nfa, dt, norm, Nfft):
    '''
    Compute scaling coefficient and critical frequency
    '''


    C = dt / (Nfa * np.pi * norm**2)
    df = 2 * np.pi / (dt * Nfft)

    return C, df

def trend_blackhar(x, win = 'blackmanharris', trend = False):
    '''
    Remove trend and compute fast fourier transform
    using Blackman-Harris window
    '''


    ## Number of data points
    N = len(x)
    # Blackman-Harris window is used
    if win == 'blackmanharris':
        window = signal.windows.blackmanharris(N)
    if win == 'hann':
        window = signal.windows.hann(N)

    ## Remove mean and compute fast fourier transform
    if trend:
        detrend = signal.detrend(x, type = 'linear')
        X = fft(window * detrend)
    else:
        X = fft(window * x)
    
    nor = norm(window, 2)
        
    ## Number of points in FFT
    Nfft = len(X)
    maxb = int(Nfft / 2 + 1)
    # Remove half of the series
    X = np.delete( X, np.arange(maxb, Nfft) )
    X[-1] = X[-1] / 2

    return X, maxb, Nfft, nor


def spectrum(x, dt, Nfa = 31, win = 'blackmanharris', trend = False):
    '''
    Frequency averaged power spectrum estimate,  GEOPHYSICAL NORMALIZATION
    Trend is removed, Blackman-Harris window is used. K.K.Kahma 1990-05-19
    
    Input data:

    x , y  = data vectors
    dt = sampling interval in seconds
    Nfa = number of elementary frequency bands which are averaged
    
    If spectrum is calculated:
        S(:,1) = f      (1/second == Hz)
        S(:,2) = Sxx    (unit*unit*second)%
    
    '''


    ## Compute dtrend, window BlackHarris, and FFT
    Xx, maxb, Nfft, nor = trend_blackhar(x, win = win, trend = trend)
    
    ## Scaling coefficient and  critical frequency 
    C, df = _C_and_df(Nfa, dt, nor, Nfft) 

    if Nfa == 1:

        ## The frequency bins (maxb)
        f = np.arange(0, maxb) * df
        print(f'The frequency bins: {len(f)}\n')
        ## The Power spectrum density
        Pxx = ( np.abs( Xx )**2 ) * C

    else:

        if Nfa > 20:

            ## When Nfa is large enough this is as fast as vectorized
            ## averaging and it requires far less memory    
            m = 0
            a = 0
            b = Nfa

            loc = int( np.fix( maxb / Nfa) )
            Pxx = np.zeros( (loc + 1) )
            f = np.zeros( (loc + 1) )

            while b <= maxb:

                Pxx[m] = np.sum( np.abs( Xx[a:b] )**2 ) * C
                f[m] = df * ( ( a + 1 + b - 2) / 2)

                a += Nfa
                b += Nfa
                m += 1


        else:

            m = int( np.fix( maxb / Nfa ) )

            Pxx = np.zeros( (m + 1) )
            f = np.zeros( (m + 1) )
            #sx = np.zeros( (m, Nfa) )

            ## The frequency bins (m)
            #for fr in range(0, m):

             #   f[fr] = ( (fr + 1) * Nfa + (-0.5 - Nfa / 2) )  * df

            f[0:m] = ( (np.arange(0, m) + 1) * Nfa + (-0.5 - Nfa / 2) )  * df

            #for ii in range(Nfa):

            #    print(f'Xx[{ii}:{m*Nfa}:{Nfa}]\n')
            #    sx[:, ii] = np.abs( Xx[ii: m*Nfa : Nfa] )**2


            sx = (np.abs( Xx[0:m*Nfa] )**2).reshape( (m, Nfa) )  
            Pxx[0:m] = np.sum( sx, axis = 1 ) * C
                

        a = m  * Nfa
        if a <= maxb:

            c = maxb + 2 - a
            Pxx[m] = np.sum( np.abs( Xx[a:maxb] )**2 ) * C * Nfa / c
            f[m] = df * ( a + 1 + maxb - 2) / 2
            #print(f'The frequency bins: {len(f)}\n')
                

    f = f / 2 / np.pi
    S = 2 * np.pi * Pxx
    Sx = np.transpose(np.array( [f, S] ))

    return f, S


def cross_spec(x, y, dt, Nfa = 31, window = 'blackmanharris', trend = False):
    '''
    Compute Cross spectrum.

    If cross spectrum is calculated:
        S(:,3) = Syy
        S(:,4) = Sxy
        S(:,5) = phase angle = 180/pi*atan2(-imag(Sxy),real(Sxy))
        S(:,6) = coherence   = abs(Sxy./sqrt(Sxx.*Syy))

    Positive phase means x leads y
    
    Elementary frequency bands 0:a0-1 (matlab index 1:a0) are ignored. 
    
    Default a0 is 0, i.e. all bands including zero (mean value) are incuded.
    '''


    ## Compute fast fourier transform of X series
    Xx, maxb, Nfft, norx = trend_blackhar(x, windows, detrend)

    ## Scaling coefficient and  critical frequency 
    C, df = _C_and_df(Nfa, dt, norx, Nfft) 

    ## Compute fast fourier transform of Y series
    Yy, maxby, Nffty, nory = trend_blackhar(y, windows, detrend)

    if Nfa == 1:

        ## The frequency bins (maxb)
        f = np.arange(0, maxb) * df

        print(f'The frequency bins: {len(f)}\n')
        ## The Power spectrum density
        Pxx = (np.abs( Xx )**2 ) * C
        Pyy = (np.abs( Yy )**2 ) * C
        Pxy = (np.conj( Xx ) * Yy ) * C
 
    else:

        if Nfa > 20:

            m = 0
            a = 0
            b = Nfa

            loc = int( np.fix( maxb / Nfa) )
            Pxx = np.zeros( (loc + 1) )
            Pyy = np.zeros( (loc + 1) )
            Pxy = np.zeros( (loc + 1) )
            f = np.zeros( (loc + 1) )
                
            while b <= maxb:

                Pxx[m] = np.sum( np.abs( Xx[a:b] )**2) * C
                Pyy[m] = np.sum( np.abs( Yy[a:b] )**2) * C
                Pxy[m] = np.sum( np.conj( Xx[a:b] ) * Yy[a:b] ) * C

                f[m] = df * ( (a + 1 + b - 2) / 2)
                
                a += Nfa
                b += Nfa
                m += 1

        else:

            m = int( np.fix( maxb / Nfa ) )

            Pxx = np.zeros( (m + 1) )
            Pyy = np.zeros( (m + 1) )
            Pxy = np.zeros( (m + 1) )
            f = np.zeros( (m + 1) )
            sx = np.zeros( (m, Nfa) )
            sy = np.zeros( (m, Nfa) ) 
            sxy = np.zeros( (m, Nfa) )

            ## The frequency bins (m)
            # for fr in range(0, m):

            #     f[fr] = ( (fr + 1) * Nfa + (-0.5 - Nfa / 2) )  * df


            f[0:m] = ( (np.arange(0, m) + 1) * Nfa + (-0.5 - Nfa / 2) )  * df

            # for ii in range(Nfa):

            #     print(f'Xx[{ii}:{m*Nfa}:{Nfa}]\n')
            #     sx[:, ii] = np.abs( Xx[ii: m*Nfa : Nfa] )**2
            #     sy[:, ii] = np.abs( Yy[ii: m*Nfa : Nfa] )**2
            #     sxy[:, ii] = np.conj( Xx[ii: m*Nfa : Nfa] ) * Yy[ii: m*Nfa : Nfa]


            sx = (np.abs( Xx[0:m*Nfa] )**2).reshape( (m, Nfa) )
            sy = (np.abs( Yy[0:m*Nfa] )**2).reshape( (m, Nfa) )
            sxy = (np.conj( Xx[0:m*Nfa] )* Yy[0:m*Nfa] ).reshape( (m, Nfa) )

            Pxx[0:m] = np.sum(sx, axis = 1) * C
            Pyy[0:m] = np.sum(sy, axis = 1) * C
            Pxy[0:m] = np.sum(sxy, axis = 1) * C
   
            a = 1 + m  * Nfa

        if a <= maxb:

            c = maxb + 1 - a
            cte = C * Nfa / c
            Pxx[m] = np.sum( np.abs( Xx[a:maxb] )**2 ) * cte
            Pyy[m] = np.sum( np.abs( Yy[a:maxb] )**2 ) * cte
            Pxy[m] = np.sum( np.conj( Xx[a:maxb] ) * Yy[a:maxb] ) * cte
            f[m] = df * ( a + maxb - 2) / 2
            print(f'The frequency bins: {len(f)}\n')

    ## Compute coherence and phase
    phase = 180 / np.pi * np.arctan2( -1 * np.imag(Pxy), np.real(Pxy) )
    coh   = np.abs( Pxy / np.sqrt( Pxx * Pyy) )
        
    f = f / 2 / np.pi
    Sx = 2 * np.pi * Pxx
    Sy = 2 * np.pi * Pyy
    Sxy = 2 * np.pi * Pxy
    S = np.transpose( np.array( [f, Sx, Sy, Sxy] ))#, phase, coh] ))

    return S, phase, coh


if __name__ == "__main__":
    pass
