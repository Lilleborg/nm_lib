#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import builtin modules
import os

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt 

def get_xx(nint: int, x0: float, xf: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an array of x values from x0 to xf with nint number of intervals between points

    Arguments:
        nint {int} -- number of intervals between grid points

    Keyword Arguments:
        xf {float} -- last value in output array (default: {10.0})
        x0 {float} -- first value in output array (default: {-4.0})

    Returns:
        tuple[np.ndarray, float] -- (the resulting x array, spacing between points x[1]-x[0])
    """
    x = np.arange(nint+1)/nint * (xf-x0) + x0
    return x, np.roll(x,-1)-x


def order_conv(hh, hh2, hh4, **kwargs) -> np.ndarray:
    """
    Computes the order of convergence of a derivative function 

    Parameters 
    ----------
    hh : `array`
        Function that depends on xx. 
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh. 
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array` 
        The order of convergence.  
    """
    return np.ma.log2((hh4[::4]-hh2[::2])/(hh2[::2]-hh))


def deriv_dnw(xx, hh, **kwargs):
    """
    Returns the downwind 2nd order derivative of hh array respect to xx array. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Keyword arguments
    -----------------
    method : 'string'
        String to indicate a particular method to use, at the moment only "roll" is recognized
        to use np.roll() for shifting indices. Using this mehtod will not loose any grid points.

    Returns
    -------
    `array`
        The downwind 2nd order derivative of hh respect to xx. First 
        grid point is ill (or missing) calculated. 
    """
    try:
        if kwargs["method"] == "roll":
            return (hh-np.roll(hh,1))/(xx - np.roll(xx,1))
        else:
            raise KeyError
    except:
        # df[i] = f[i] - f[i-1] /dx
        # dx[i] = x[i] - x[i-1]
        # set i = 1 -> df = f[1] - f[0]
        #              dx = x[1] - x[0]
        return (hh[1:] - hh[:-1])/(xx[1:]-xx[:-1])


def deriv_upw(xx, hh, **kwargs):
    """
    returns the upwind 2nd order derivative of hh respect to xx. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Keyword arguments
    -----------------
    method : 'string'
        String to indicate a particular method to use, at the moment only "roll" is recognized
        to use np.roll() for shifting indices. Using this mehtod will not loose any grid points.

    Returns
    ------- 
    `array`
        The upwind 2nd order derivative of hh respect to xx. Last 
        grid point is ill calculated. 
    """
    try:
        if kwargs["method"] == "roll":
            return (np.roll(hh,-1)-hh)/(np.roll(xx,-1)-xx)
        else:
            raise KeyError
    except:
        # df[i] = f[i+1] - f[i] /dx
        # dx[i] = x[i+1] - x[i]
        # set i = 0 -> df = f[1] - f[0]
        #              dx = xx[1] - x[0]
        return (hh[1:] - hh[:-1])/(xx[1:]-xx[:-1])


def deriv_cent(xx, hh, **kwargs):
    """
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Keyword arguments
    -----------------
    method : 'string'
        String to indicate a particular method to use, at the moment only "roll" is recognized
        to use np.roll() for shifting indices. Using this mehtod will not loose any grid points.

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """
    try:
        if kwargs["method"] == "roll":
            return (np.roll(hh,-1) - np.roll(hh,1))/(2*(np.roll(xx,-1)-xx))
        else:
            raise KeyError
    except:
        # df[i] = f[i+1]-f[i-1] / 2dx
        # dx[i] = x[i+1]-x[i-1]
        # set i = 1 -> df = f[2] - f[0]
        return (hh[2:] - hh[:-2])/(2*(xx[2:]-xx[1:-1]))


def deriv_4tho(xx, hh, **kwargs): 
    """
    Returns the 4th order derivative of hh respect to xx.

    Parameters 
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Keyword arguments
    -----------------
    method : 'string'
        String to indicate a particular method to use, at the moment only "roll" is recognized
        to use np.roll() for shifting indices. Using this mehtod will not loose any grid points.

    Returns
    -------
    `array`
        The centered 4th order derivative of hh respect to xx. 
        Last and first two grid points are ill calculated. 
    """
    try:
        if kwargs["method"] == "roll":
            return (-np.roll(hh,-2) + 8*np.roll(hh,-1) - 8*np.roll(hh,1) + np.roll(hh,2))/(12*(np.roll(xx,-1) - xx))
        else:
            raise KeyError
    except:
        # df[i] = - f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2] / 12dx
        # dx[i] = x[i+1] - x[i]
        # set i = 2 -> df = f[4] - 8f[3] + 8f[1] - f[0]
        #              dx = x[3] - x[2]
        dh4th   = -hh[4:] + 8*hh[3:-1] - 8*hh[1:-3] + hh[:-4]
        dx4th   = xx[3:-1] - xx[2:-2]
        return (dh4th)/(12*dx4th)
   

def step_adv_burgers(xx, hh, a, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_upw(x, y, method="roll"), **kwargs) -> tuple[float,np.ndarray]: 
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 

    Requires 
    ---------- 
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default clf_cut=0.98. 
    ddx : `lambda function`
        Allows to select the type of spatial derivative. 
        By default lambda x,y: deriv_upw(x, y, method="roll)

    Returns
    -------
    `tuple`
        1) `float`: Time interval.
        2) `array`: Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    dt = cfl_cut*cfl_adv_burger(a, xx)
    rhs = -a*ddx(xx,hh)
    # NB! Mind boundaries of the spatial derivative term !!!
    if "bnd_limits" in kwargs:
        low, up = kwargs["bnd_limits"]
        up = None if up == 0 else -up
        rhs = rhs[low:up]
        bnd_type = kwargs["bnd_type"] if "bnd_type" in kwargs else "wrap"
        rhs = np.pad(rhs, kwargs["bnd_limits"], bnd_type)
    return dt, rhs


def cfl_adv_burger(a,x): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the advective term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    ------- 
    `float`
        min(dx/|a|)
    """
    dx = (np.roll(x,-1) - x)[:-1]   # exlude the last ill calcullated value
    return np.min(dx/np.abs(a))


def evolv_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    """
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).  
    bnd_type : `string`
        Allows to select the type of boundaries. 
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1].

    Returns
    ------- 
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    #step_adv_burgers -> rhs 
    #fix boundaries from the rhs 
    #compute ut+1
    # print("a evolv", a)
    tt = np.zeros(nt)
    uunt = np.zeros((nt,len(hh)))
    uunt[0,:] = hh
    for n in range(nt-1):
        dt, step = step_adv_burgers(xx, uunt[n,:], a, cfl_cut=cfl_cut, ddx=ddx, bnd_limits=bnd_limits, bnd_type=bnd_type)
        uunt[n+1,:] = uunt[n,:] + step * dt
        tt[n+1] = tt[n] + dt
    return tt, uunt


def evolv_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger. 
        By default 0.98.
    ddx : `lambda function` 
        Allows to change the space derivative function. 
    bnd_type : `string` 
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array` 
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def evolv_Lax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def evolv_Lax_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def step_uadv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.  

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """       


def cfl_diff_burger(a,x): 
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the diffusive term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    -------
    `float`
        min(dx/|a|)
    """


def ops_Lax_LL_Add(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Additive Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def ops_Lax_LL_Lie(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def ops_Lax_LL_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme. Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)` 
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """



def osp_Lax_LH_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Strang Operator Splitting scheme. One step is with a Lax method 
    and the second step is the Hyman predictor-corrector scheme. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def step_diff_burgers(xx, hh, a, ddx = lambda x,y: deriv_cent(x, y), **kwargs): 
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that 
    depends on xx. 
    
    Parameters
    ----------    
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """    


def NR_f(xx, un, uo, a, dt, **kwargs): 
    r"""
    NR F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    


def jacobian(xx, un, a, dt, **kwargs): 
    r"""
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """    


def Newton_Raphson(xx, hh, a, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    NR scheme for the burgers equation. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian(xx, ug, a, dt) # Jacobian 
            ff1=NR_f(xx, ug, uo, a, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error: 
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) # error
            #err = np.max(np.abs(un-ug))
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt



def NR_f_u(xx, un, uo, dt, **kwargs): 
    r"""
    NR F function.

    Parameters
    ----------  
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    


def jacobian_u(xx, un, dt, **kwargs): 
    """
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """    


def Newton_Raphson_u(xx, hh, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float` 
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]        

    Returns
    -------
    t : `array`
        Time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `array(int)` 
        Number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian_u(xx, ug, dt) # Jacobian 
            ff1=NR_f_u(xx, ug, uo, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) 
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt

def taui_sts(nu, niter, iiter): 
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------   
    nu : `float`
        Coefficient, between (0,1).
    niter : `int` 
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float` 
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """

def evol_sts(xx, hh, nt,  a, cfl_cut = 0.45, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], nu=0.9, n_sts=10): 
    """
    Evolution of the STS method. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.45
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations. 
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def hyman(xx, f, dth, a, fold=None, dtold=None,
        cfl_cut=0.8, ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if (np.any(fold) == None):
        firstit=False
        fold = np.copy(f)
        f = (np.roll(f,1)+np.roll(f,-1))/2.0 + u1_temp * dth 
        dtold=dth

    else:
        ratio = dth/dtold
        a1 = ratio**2
        b1 =  dth*(1.0+ratio   )
        a2 =  2.*(1.0+ratio    )/(2.0+3.0*ratio)
        b2 =  dth*(1.0+ratio**2)/(2.0+3.0*ratio)
        c2 =  dth*(1.0+ratio   )/(2.0+3.0*ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)
        
        if bnd_limits[1]>0: 
            u1_c =  f[bnd_limits[0]:-bnd_limits[1]]
        else: 
            u1_c = f[bnd_limits[0]:]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1]>0: 
        u1_c = f[bnd_limits[0]:-bnd_limits[1]]
    else: 
        u1_c = f[bnd_limits[0]:]
    f = np.pad(u1_c, bnd_limits, bnd_type)
    
    dtold=dth

    return f, fold, dtold


def hyman_corr(f, fsav, dfdt, c2):

    return  fsav  + c2* dfdt


def hyman_pred(f, fold, dfdt, a1, b1, a2, b2): 

    fsav = np.copy(f)
    tempvar = f + a1*(fold-f) + b1*dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2*(fsav-tempvar) + b2*dfdt    
    f = tempvar
    
    return f, fold, fsav

