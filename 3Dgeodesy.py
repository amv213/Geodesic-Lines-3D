"""
GEODESIC LINES SCRIPT

Inspired by http://www.physikdidaktik.uni-karlsruhe.de/software/geodesiclab/a3.html

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits import mplot3d


def f(x, y):
    """
    Function on which to calculate geodesic lines.
    Here it's a rescaled Gaussian but can change it to whatever you like

    args:
        x:      x coordinate of the point
        y:      y coordinate of the point

    returns: f(x,y)

    """

    # Flat Bump
    #z = .03 / (.1 + x * x + y * y)

    # Medium Bump
    #z = .3 / (.3 + x**2 + y**2)

    # High bump
    #z = .2 / (.1 + x**2 + y**2)

    # Crater
    #z = (x**2 + y**2) * (1 - x**2 - y**2)

    # Bump and dimple
    #z = .1 / (.1 + (y - .5)**2 + x**2) - .1 / (.1 + (y + .5)**2 + x**2)

    # Hilly Landscape
    z = np.sin(5*y) * np.cos(5*x) / 2 / (.5 + x**2 + y**2)

    return z


def dfdx(x, y, epsilon=0.0001):
    """
    Approximate derivative of the function with respect to x, evaluated at the point (x,y)

    args:
        x:          x coordinate of the point
        y:          y coordinate of the point
        epsilon:    infinitesimal epsilon for interval on which to approximate derivative

    returns: df/dx
    """

    return (f(x + epsilon, y) - f(x - epsilon, y))/(2*epsilon)


def dfdy(x, y, epsilon=0.0001):
    """
    Approximate derivative of the function with respect to y, evaluated at the point (x,y)

    args:
        x:          x coordinate of the point
        y:          y coordinate of the point
        epsilon:    infinitesimal epsilon for interval on which to approximate derivative

    returns: df/dy
    """

    return (f(x, y + epsilon) - f(x, y - epsilon)) / (2 * epsilon)


def fnormal(x, y):
    """
     Normal vector, evaluated at the point (x,y)

     args:
         x:          x coordinate of the point
         y:          y coordinate of the point

     returns: vector
     """

    n = np.array([-dfdx(x,y), -dfdy(x,y), 1])

    # We return the normalised vector
    return n/np.sqrt(n[0]**2 + n[1]**2 + 1)


def is_out_bounds(x, y, x_ax_min, x_ax_max, y_ax_min, y_ax_max):
    """
    Checks if a given point is outside of the plotting area.

    args:
        x:      x coordinate of the point
        y:      y coordinate of the point
        *:      x/y min/max boundaries of the plotting area

    returns: boolean
    """

    # Our plotting area is hard coded to span the (-1, 1)^2 range
    if x < x_ax_min or x > x_ax_max or y < y_ax_min or y > y_ax_max:
        return 1
    else:
        return 0


# ACTUAL SCRIPT STARTS HERE:
if __name__ == "__main__":

    # DEFINE GENERAL PARAMETERS

    x_ax_min, x_ax_max = -1, 1      # x-axis limits
    y_ax_min, y_ax_max = -1, 1      # y-axis limits
    smoothness = 200                # smoothness of the surface plot
    max_num_iter = 1000              # max number of points per line
    step_size = 0.01                # arbitrary, increase to do bigger steps on each iteration
    num_lines = 28                  # number of geodesic lines to draw
    cmap = cm.get_cmap('hsv')   # decide which colormap to use for the geodesic lines

    # PLOT SURFACE:

    fig = plt.figure(figsize=(32, 16))

    # fig3D = plt.figure(figsize=(16, 16))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax3.set_axis_off()

    # fig2D = plt.figure(figsize=(16, 16))
    #ax2 = plt.gca()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect('equal')
    ax2.set_axis_off()

    x = np.linspace(x_ax_min, x_ax_max, smoothness)
    y = np.linspace(y_ax_min, y_ax_max, smoothness)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    surf3 = ax3.plot_surface(X, Y, Z, cmap='binary', vmin=np.min(Z), vmax=4*np.max(Z))  # 3D
    surf2 = ax2.contourf(X, Y, Z, cmap='binary', vmin=np.min(Z), vmax=4*np.max(Z))  # 2D
    cb = fig.colorbar(surf2, shrink=0.75)

    # PLOT GEODESIC LINES:

    # Here we are simply starting the lines from different starting points
    for (l, x_start) in enumerate(np.linspace(x_ax_min+0.055, x_ax_max-0.055, num_lines)):

        # starting point
        x0 = x_start
        y0 = y_ax_min

        # initial step
        dx0 = 0.
        dy0 = 0.05

        # Normalised learning rate
        c = step_size
        eta = c / np.sqrt(dx0**2 + dy0**2)

        # Initialize position arrays
        x = np.array([x0, x0 + eta*dx0])
        y = np.array([y0, y0 + eta*dy0])

        # MAIN ALGORITHM I
        for i in range(1, max_num_iter):  # we simulate N time-steps

            # Current position
            xt = x[i]
            yt = y[i]
            ft = f(xt, yt)

            # If the current point is out of the plotting area we stop
            if is_out_bounds(xt, yt, x_ax_min, x_ax_max, y_ax_min, y_ax_max):

                print(f"Geodesic Line {l} Finished")
                break

            # Previous position
            xtm1 = x[i-1]  # read as 'x_{t-1}'
            ytm1 = y[i-1]
            ftm1 = f(xtm1, ytm1)

            # Symmetric position wrt current position
            xsymp = xt + (xt - xtm1)  # read as 'x symmetric +'
            ysymp = yt + (yt - ytm1)
            fsymp = ft + (ft - ftm1)

            # Difference between predicted 'symmetric' value of f and actual value of f at the symmetric point
            df = fsymp - f(xsymp, ysymp)
            # Normal at current position
            n = fnormal(xt, yt)
            # Correction factor
            gamma = df*n[2]

            # Next position. We are essentialy correcting the 'symmetric' guess by a factor gamma*n
            xtp1 = xsymp - gamma*n[0]  # read as 'x_{t+1}'
            ytp1 = ysymp - gamma*n[1]

            # Update our lists with the new position
            x = np.append(x, xtp1)
            y = np.append(y, ytp1)

        # x and y now hold complete trajectory (based on some kind of iterative midpoint method?)

        # MAIN ALGORITHM II
        # This seems to be doing a second pass through our trajectories and tweaking them to get final results

        posx = np.array(len(x))  # initialize empty arrays
        posy = np.array(len(y))
        posz = np.array(len(x))

        for i in range(len(x)-1):

            # Current position
            xt = x[i]
            yt = y[i]

            # Offset from next position
            dx = x[i+1] - xt
            dy = y[i+1] - yt

            dn = np.sqrt(dx**2 + dy**2)
            df = dfdx(xt, yt)*dx/dn + dfdy(xt, yt)*dy/dn
            dfn = 1/np.sqrt(1 + df**2)

            # Correct current position

            if i % 2 == 1:  # not too sure of this step but it's what the website was doing ...
                pm = 1      # ... every second point we will be switching between adding / subtracting the correction
            else:
                pm = -1

            xt_actual = xt + pm*dfn*dx  # pm is 'plus or minus'
            yt_actual = yt + pm*dfn*dy
            zt_actual = f(xt_actual, yt_actual)

            # Push to lists
            posx = np.append(posx, xt_actual)
            posy = np.append(posy, yt_actual)
            posz = np.append(posz, zt_actual)

        # Plot trajectory
        ax3.plot(posx[1:], posy[1:], posz[1:], c=cmap(l/num_lines), zorder=3)  # 3D
        ax2.plot(posx[1:], posy[1:], c=cmap(l/num_lines), zorder=3)  # 2D
        # (omit first point because the algorithm messes it up for some reason)


    plt.xlim(x_ax_min, x_ax_max)
    plt.ylim(y_ax_min, y_ax_max)

    plt.tight_layout()
    plt.show()
