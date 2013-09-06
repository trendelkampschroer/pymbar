from numpy import *
from scipy.signal import fftconvolve
from scipy.optimize import newton

def __convert_to_timeseries(X):
    if(X.ndim>2):
        raise ValueError("Can not convert to 2d time-series")
    elif(X.ndim==1):
        N=X.shape[0]
        return X.reshape(N, 1)
    else:
        return X

def fftcorrelate(x, y, mode='valid'):
    """
    Compute the discrete time correlation 

    C[x, y](...,k_i,...)= \sum_{n_i=-\infty}^{\infty} x_{n_i+k_i} y_{n_i}

    using fast fourier transformation (fft).

    Parameters :
    ------------
    x : array_like
        First input
    y : array_like
         Second input. Should have the same number of dimensions as `in1`;
         if sizes of `in1` and `in2` are not equal then `in1` has to be the
         larger array.

    mode : str {'full', 'valid', 'same'}, optional   

        A string indicating the size of the output:

        ''full''
            The output is the full discrete linear convolution 
            of the inputs. (Default)
        ''valid''
            The output consists only of those elements that do
            not rely on the zero-padding.
        ''same''
            The output is the same size as in1, centered with
            respect to the 'full' output.
               
    """
    idy=[slice(None, None, -1) for i in range(y.ndim)]
    return fftconvolve(x, y[idy], mode=mode)

def autocorrelation(data):
    """
    Estimate the autocorrelation function for a time-series
    with d-dimensional observations.

    Parameters :
    ------------
    data : ndarray, shape=(N, d)
        The time-series data. Each row contains a single observation
        with columns containing individual coordinates.

    Returns :
    ---------
    acorr : ndarray, shape=(N/2,)
        The autocorrelation function at time lags 0,...,N/2
           
    """
    if(data.ndim!=2):
        raise ValueError("Can only handle 2d arrays")
    else:        
        N=data.shape[0]
        d=data.shape[1]
        t_max=N/2
        mu=mean(data, axis=0)
        sigma=std(data, axis=0)
        X=data-mu
        acorr=zeros(t_max)
        for i in range(d):
            acorr+=fftcorrelate(X[:, i], X[:, i], mode='full')[N-1:(N-1)+t_max]/(N-arange(t_max))
        acorr=acorr/dot(sigma, sigma)
        return acorr

def correlation_time_trapezoidal(data):
    """
    Estimate the correlation time t_corr for a given time-series.

    The correlation time is estimated in three steps:
    
    i) Compute the auocorrelation function C(\tau) at different time lags
        using fft.
    ii) Identify the first t_0 for which C(\tau)<0.
    iii) Estimate the correlation time 

        t_corr \approx \int_{0}^{t_0} d\tau C(\tau)

        using the trapezoidal rule for the estimated correlation function
        values up to t_0.

    Parameters :
    ------------
    data : array_like
        The time series data. The data points should be recored at equally
        spaced time points,

        data=(..X(t_i), X(t_{i+1}),...) with t_{i+1}-t_i=t_{j+1}=t_{j} \for all i,j

    Returns :
    ---------
    tcorr : float
        The correlation time     

    # """
    # N=len(data)
    # t_max=N/2

    # """Shift and rescale data"""
    # mu=mean(data)
    # sigma=std(data)
    # X=(data-mu)/sigma

    # """Compute the correlation function at \tau=0...\tau=t_max"""
    # corr=fftcorrelate(X, X, mode='full')[N:2*N-t_max]/(N-arange(t_max))    

    data=__convert_to_timeseries(data)
    
    """Compute autocorrelation function for \tau=0,...,N/2"""
    acorr=autocorrelation(data)

    """Identify t_0"""
    t_0=min(where(acorr<0.0)[0])

    """Restrict to [0, t_0)"""
    acorr=acorr[0:t_0]
    tcorr=0.5*sum(acorr[0:-1]+acorr[1:])
    return tcorr

def correlation_time(data):
    """
    Estimate the correlation time t_corr for a given N-dimensional array.

    The correlation time is estimated in three steps:
    
    i) Compute the auocorrelation function C(\tau) at different time lags
        using fft.
    ii) Identify the first \tau_0 for which C(\tau)<0.
    iii) Estimate the value 

        I=\int_{0}^{\tau_0} d\tau C(\tau)

        using the trapezoidal rule for the estimated correlation function
        values up to \tau_0. 
    iv) Use the Newton-Rhapson method to estimate the correlation time
        tcorr as zero point of the function

        f(x)=x-x*e**(-t_0/x)-I

        with derivative

        f'(x)=1-e**(-t_0/x)*(1+t_0/x)

    Parameters :
    ------------
    data : array_like
        The time series data. The data points should be recored at equally
        spaced time points,

        data=(..X(t_i), X(t_{i+1}),...) with t_{i+1}-t_i=t_{j+1}=t_{j} \for all i,j

    Returns :
    ---------
    tcorr : float
        The correlation time     

    """
    data=__convert_to_timeseries(data)
    
    """Compute autocorrelation function for \tau=0,...,N/2"""
    acorr=autocorrelation(data)

    """Identify t_0"""
    t_0=min(where(acorr<0.0)[0])

    """Restrict to [0, t_0)"""
    acorr=acorr[0:t_0]

    """Use trapezoidal rule to compute integral under autocorrelation curve"""
    I=.5*sum(acorr[0:-1]+acorr[1:])

    """Use Newton method with I as initial guess to compute correlation time"""
    tcorr=newton(__f, I, fprime=__D1f, args=(t_0, I))
    return tcorr
    
def __f(x, t_0, I):
    return x-x*e**(-t_0/x)-I

def __D1f(x, t_0, I):
    return 1-e**(-t_0/x)*(1+t_0/x)

def statistical_inefficiency(data):
    """
    Compute the statistical inefficiency g of a given time-series,

        g = 1 + 2*tcorr
    
    with tcorr the correlation time of the time-series.     

    Parameters :
    ------------
    data : array_like
        The time series data. The data points should be recored at equally
        spaced time points,

        data=(..X(t_i), X(t_{i+1}),...) with t_{i+1}-t_i=t_{j+1}=t_{j} \for all i,j

    Returns :
    ---------
    g : float
        Statistical inefficiency

    """
    tcorr=correlation_time(data)
    g=1.0+2.0*tcorr
    return g

def subsample_indices(data, g=None, uniform=False):
    """
    Compute indices of uncorrelated samples in data.

    Parameters :
    ------------
    data : array_like
        The time series data. The data points should be recored at equally
        spaced time points,

        data=(..X(t_i), X(t_{i+1}),...) with t_{i+1}-t_i=t_{j+1}=t_{j} \for all i,j

    g : float (optional)
        Statistical inefficiency

    Returns :
    ---------
    indices : ndarray, shape(k, )
        List of integer indices.       
    
    """
    N=data.shape[0]
        
    if not g:
        g=statistical_inefficiency(data)

    if uniform:
        dt=int(ceil(g))
        k=int(floor(N/dt))
        indices=arange(0, N, dt)
        return indices
    else:
        dt=g
        float_indices=arange(0.0, 1.0*N-1.0, dt)
        indices=float_indices.round().astype(int)
        return indices
        
def subsample(data, g=None, uniform=False):
    """
    Subsample correlated time-series data

    Parameters :
    ------------
    data : array_like
        The time series data. The data points should be recored at equally
        spaced time points,

        data=(..X(t_i), X(t_{i+1}),...) with t_{i+1}-t_i=t_{j+1}=t_{j} \for all i,j

    g : float (optional)
        Statistical inefficiency

    Returns :
    ---------
    data_sub : array_like
        Uncorrelated subset of the input data.      
    
    """
    indices=subsample_indices(data, g=g, uniform=uniform)
    return data[indices]
