"""
PYTHON GEODESIC LINES SCRIPT

Inspired by http://www.physikdidaktik.uni-karlsruhe.de/software/geodesiclab/a3.html

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits import mplot3d


class Canvas:
    """
    The canvas area containing the surface function on which we want to compute geo-lines.
    """

    def __init__(self, f, xmin=-1, xmax=1, ymin=-1, ymax=1, smoothness=200):

        self.f = f  # f(x,y) function defining the 'world' landscape

        self.xmin = xmin    # lower x-axis limit of the grid on which to evaluate the function
        self.xmax = xmax    # higher x-axis limit of the grid on which to evaluate the function
        self.ymin = ymin    # lower y-axis limit of the grid on which to evaluate the function
        self.ymax = ymax    # higher y-axis limit of the grid on which to evaluate the function

        self.smoothness = smoothness    # number of points over which to evaluate the function

    def dfdx(self, x, y):
        """
        Approximate derivative of the function with respect to x, evaluated at the point (x,y)

        args:
            x:          x coordinate of the point
            y:          y coordinate of the point

        returns: df/dx
        """

        epsilon = 0.0001
        return (self.f(x + epsilon, y) - self.f(x - epsilon, y)) / (2 * epsilon)

    def dfdy(self, x, y):
        """
        Approximate derivative of the function with respect to y, evaluated at the point (x,y)

        args:
            x:          x coordinate of the point
            y:          y coordinate of the point

        returns: df/dy
        """

        epsilon = 0.0001
        return (self.f(x, y + epsilon) - self.f(x, y - epsilon)) / (2 * epsilon)

    def fn(self, x, y):
        """
         Unit normal vector, evaluated at the point (x,y)

         args:
             x:          x coordinate of the point
             y:          y coordinate of the point

         returns: unit normal vector
         """

        n = np.array([self.dfdx(x, y), self.dfdy(x, y), -1])
        mag = np.sqrt(n[0] ** 2 + n[1] ** 2 + 1)

        return n / mag


class GeoLine:
    """
    The geodesic line object and all its properties
    """

    def __init__(self, canvas, x0, y0, dx0=0, dy0=0.05, step_size=0.01):

        self.canvas = canvas    # the Canvas object on which the geo-line will be evaluated

        self.x = np.array([x0])     # array of x coordinates of the geo-line trajectory
        self.y = np.array([y0])     # array of y coordinates of the geo-line trajectory
        self.z = self.canvas.f(self.x, self.y)  # array of z coordinates of the geo-line trajectory

        self.init_step(x0, y0, dx0, dy0, step_size)     # do arbitrary initialisation step in one direction

        self.finished = False   # whether the whole geo-line trajectory has been calculated

    def init_step(self, x, y, dx, dy, step_size):
        """
        Perform a first initialisation step to prepare trajectory for iterative algorithm

        args:
            x:          current geo-line x position
            y:          current geo-line y position
            dx:         initialisation step in the x direction
            dy:         initialisation step in the y direction
            step_size:  hyperparameter regulating scale of initialisation step

        returns:    updated geo-line position arrays
        """

        eta = step_size / np.sqrt(dx ** 2 + dy ** 2)

        self.x = np.append(self.x, x + eta * dx)
        self.y = np.append(self.y, y + eta * dy)
        self.z = np.append(self.z, self.canvas.f(self.x[-1], self.y[-1]))

    def is_out_bounds(self, x, y):
        """
         Checks if a given point is outside of the canvas area.

         args:
             x:      x coordinate of the point
             y:      y coordinate of the point

         returns: updated geo-line status
         """

        if x < self.canvas.xmin or x > self.canvas.xmax or y < self.canvas.ymin or y > self.canvas.ymax:
            self.finished = True

    def step_next(self):
        """
        Advances the geo-line by one step.
        Calculation based on geo-line position at time t and at time (t-1)

        returns: updated geo-line position arrays and updated .finished status
        """

        # Current position
        xt = self.x[-1]
        yt = self.y[-1]
        zt = self.canvas.f(xt, yt)

        # Previous position
        xtm1 = self.x[-2]
        ytm1 = self.y[-2]
        ztm1 = self.canvas.f(xtm1, ytm1)

        # Symmetric position wrt current position
        # i.e. position such that position at time t is the midpoint between the position at (t-1) and this position
        xp = xt + (xt - xtm1)  # read as 'x+'
        yp = yt + (yt - ytm1)
        zp = zt + (zt - ztm1)

        # Difference between predicted value of f and actual value of f at the symmetric point
        dz = zp - self.canvas.f(xp, yp)
        # Normal at current position
        n = self.canvas.fn(xt, yt)
        # Correction factor
        gamma = dz * n[2]

        # Next position (correcting the 'symmetric' guess by a factor gamma*n)
        xtp1 = xp - gamma * n[0]  # read as 'x_{t+1}'
        ytp1 = yp - gamma * n[1]

        # Update geo-line trajectory with the new position
        self.x = np.append(self.x, xtp1)
        self.y = np.append(self.y, ytp1)
        self.z = np.append(self.z, self.canvas.f(xtp1, ytp1))

        # Check if new step is out of bounds, then geo-line is finished
        self.is_out_bounds(xtp1, ytp1)


def my_func(x, y):
    """
    Landscape on which to calculate geodesic lines.

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


if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------------------------------
    # ALGORITHM
    # ------------------------------------------------------------------------------------------------------------------

    max_num_iter = 1000     # max number of points per line
    num_lines = 28          # number of geodesic lines to draw

    # -- SANDBOX INITIALIZATION
    # -- The surface on which we want to calculate geo-lines

    sandbox = Canvas(my_func)  # can use any arbitrary function

    # -- CALCULATE GEO-LINES

    buffer_x, buffer_y, buffer_z = [], [], []  # initialize containers for geo-line trajectories

    # creating multiple uniformly spaced geo-lines all starting on the canvas y-edge
    for (line_idx, x_start) in enumerate(np.linspace(sandbox.xmin+0.055, sandbox.xmax-0.055, num_lines)):

        # Initialize geo-line, evolving in our sandbox
        geoline = GeoLine(sandbox, x0=x_start, y0=sandbox.ymin)

        # Propagate geo-line for N time-steps
        for i in range(1, max_num_iter):

            if geoline.finished:
                break
            else:
                geoline.step_next()

        print(f"Geodesic Line {line_idx} | DONE")

        buffer_x.append(geoline.x), buffer_y.append(geoline.y), buffer_z.append(geoline.z)  # update buffers

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------------------------------------------------------

    # -- PLOT SANDBOX:
    fig = plt.figure(figsize=(32, 16))

    x = np.linspace(sandbox.xmin, sandbox.xmax, sandbox.smoothness)
    y = np.linspace(sandbox.ymin, sandbox.ymax, sandbox.smoothness)
    X, Y = np.meshgrid(x, y)
    Z = sandbox.f(X, Y)

    # ---- 3D SANDBOX
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax3.set_axis_off()

    surf3 = ax3.plot_surface(X, Y, Z, cmap='binary', vmin=np.min(Z), vmax=4 * np.max(Z))

    # ---- 2D SANDBOX
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect('equal')
    ax2.set_axis_off()

    surf2 = ax2.contourf(X, Y, Z, cmap='binary', vmin=np.min(Z), vmax=4 * np.max(Z))
    cb = fig.colorbar(surf2, shrink=0.75)

    # -- PLOT GEO-LINES
    cmap = cm.get_cmap('hsv')  # decide which colormap to use for the geodesic lines
    for i in range(num_lines):

        ax2.plot(buffer_x[i], buffer_y[i], c=cmap(i / num_lines), zorder=3)  # Plot 2D trajectory
        ax3.plot(buffer_x[i], buffer_y[i], buffer_z[i], c=cmap(i / num_lines), zorder=3)  # Plot 3D trajectory

    plt.xlim(sandbox.xmin, sandbox.xmax)
    plt.ylim(sandbox.ymin, sandbox.ymax)

    plt.tight_layout()
    plt.show()
