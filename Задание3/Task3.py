# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ffc import *
# from fenics import *
from dolfin import *
from mshr import *

num_fragments = 100
R = 1
alfa = 5

tol = 1E-14

T = 2.0
num_steps = 10
dt = T / num_steps
t = np.zeros(num_steps + 1)
for n in range(num_steps + 1):
    t[n] = n * dt

def boundary_value_problem(h_val, g_val, f_val, txt, formula):
    def boundary_D(x, on_boundary):
        if near(x[0]**2 + x[1]**2, R**2, tol) and x[1] > 0.0:
            return True
        else:
            return False

    domain = Circle(Point(0, 0), R, num_fragments)
    mesh = generate_mesh(domain, num_fragments, "cgal")

    V = FunctionSpace(mesh, 'P', 1)
    h = Expression(h_val, degree=2)
    g = Expression(g_val, R = R, degree=2)
    f = Expression(f_val, alfa = alfa, degree=2)

    bc = DirichletBC(V, h, boundary_D)

    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = (dot(grad(u), grad(v)) + alfa*u*v)*dx
    L = f*v*dx + g*v*ds
    u = Function(V)
    solve(a == L, u, bc)

    error_L2 = errornorm(h, u, 'L2')
    print("L2-error = ", error_L2)

    vertex_values_u_D = h.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))
    print("C-error = ", error_C)

    Plot(mesh, u, h, txt, formula, error_L2, error_C)

def Plot(mesh, u, h, txt, formula, error_L2, error_C):
    fig = plt.figure()
    fig.suptitle("L2-error = " + str(error_L2) + "\n" + "C-error = " + str(error_C))
    fig.set_size_inches(20, 10)
    ax = fig.add_subplot(121)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    ax1 = fig.add_subplot(122)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', '5%', '5%')

    fig.subplots_adjust(wspace=0.5)

    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    ax.set_aspect('equal')
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    tpc = ax.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    cax.cla()
    fig.colorbar(tpc, cax=cax)

    ax.set_title(r"Numerical solution" + "\n" + formula)
    ax1.set_title(r"Analytical solution" + "\n" + formula)

    ax1.set_aspect('equal')
    zfaces = np.asarray([h(cell.midpoint()) for cell in cells(mesh)])
    tpc1 = ax1.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    cax1.cla()
    fig.colorbar(tpc1, cax=cax1)
    plt.savefig('/mnt/e/Python/Задание3/boundary_value_problem/sol_' + txt + '.png')


fig = plt.figure()
fig.set_size_inches(20, 10)
ax = fig.add_subplot(121)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

ax1 = fig.add_subplot(122)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '5%', '5%')

fig.subplots_adjust(wspace=0.5)

def heat_conduction_problem(h_val, g_val, f_val, txt, formula):
    def boundary_D(x, on_boundary):
        if near(x[0]**2 + x[1]**2, R**2, tol) and x[1] > 0.0:
            return True
        else:
            return False

    def plot_gif_numerical_sol(n):
        ax.clear()
        ax1.clear()
        
        h.t = t[n + 1]
        g.t = t[n + 1]
        f.t = t[n + 1]

        solve(a == L, u, bc)

        u_e = interpolate(h, V)

        k = mesh.num_vertices()
        d = mesh.geometry().dim()
        mesh_coordinates = mesh.coordinates().reshape((k, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

        ax.set_aspect('equal')
        zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
        img = ax.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
        cax.cla()
        fig.colorbar(img, cax=cax)

        ax1.set_aspect('equal')
        zfaces = np.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
        img1 = ax1.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
        cax1.cla()
        fig.colorbar(img1, cax=cax1)

        ax.set_title(r"Numerical solution" + "\n" + formula)
        ax1.set_title(r"Analytical solution" + "\n" + formula)

        # fig.savefig('/mnt/e/Python/Задание3/heat_conduction_problem/numerical_sol_' + str(n) + '.png')

        error_L2[n + 1] = errornorm(u_e, u, 'L2')
        print('t = %.2f: error_L2 = %g' % (t[n + 1], error_L2[n]))

        vertex_values_u_D = u_e.compute_vertex_values(mesh)
        vertex_values_u = u.compute_vertex_values(mesh)
        error_C[n + 1] = np.max(np.abs(vertex_values_u - vertex_values_u_D))
        print('t = %.2f: C-error = %g' % (t[n + 1], error_C[n]))

        u_n.assign(u)

        return img, img1

    domain = Circle(Point(0, 0), R, num_fragments)
    mesh = generate_mesh(domain, num_fragments, "cgal")

    V = FunctionSpace(mesh, 'P', 1)
    h = Expression(h_val, degree=2, t=0)
    g = Expression(g_val, degree=2, t=0, R=R)
    f = Expression(f_val, alfa = alfa, t=0, degree=2)

    bc = DirichletBC(V, h, boundary_D)

    u_n = interpolate(h, V)

    u = TrialFunction(V)
    v = TestFunction(V)

    F = u*v*dx + alfa*dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx - alfa*dt*g*v*ds
    a, L = lhs(F), rhs(F)

    u = Function(V)
    error_L2 = np.zeros(num_steps + 1)
    error_C = np.zeros(num_steps + 1)

    numerical_sol = animation.FuncAnimation(fig, 
                                      plot_gif_numerical_sol, 
                                      frames=num_steps,
                                      interval = 1000,
                                      repeat = True)

    numerical_sol.save('/mnt/e/Python/Задание3/heat_conduction_problem/numerical_sol_' + txt + '.gif',
                    writer='imagemagick')

    fig_err, ax_err = plt.subplots()
    ax_err.plot(t, error_L2, 'b', label='error_L2')

    ax_err.plot(t, error_C, 'r', label='error_C')
    ax_err.legend(loc='best')
    ax_err.set_title(r"" + formula)
    plt.savefig('/mnt/e/Python/Задание3/heat_conduction_problem/error_graph_' + txt + '.png')


def main():
    # boundary_value_problem('pow(x[0], 2) + pow(x[1], 2)', '2.0*pow(x[0], 2)/R + 2.0*pow(x[1], 2)/R', '-4.0 + alfa*(pow(x[0], 2) + pow(x[1], 2))', 'u1', \
    #                         '$x^{2}+y^{2}$')
    # boundary_value_problem('sin(x[0]) + cos(x[1])', 'cos(x[0])*x[0]/R - sin(x[1])*x[1]/R', '(1.0 + alfa) * (sin(x[0]) + cos(x[1]))', 'u2', \
    #                         '$\sin{x}+\cos{y}$')
    # boundary_value_problem('exp(pow(x[0], 2)) + exp(pow(x[1], 2))', 'exp(pow(x[0], 2))*2.0*pow(x[0], 2)/R + exp(pow(x[1], 2))*2.0*pow(x[1], 2)/R', \
    #                     '-(exp(pow(x[0], 2)) * (4.0*pow(x[0], 2) + 2.0) + exp(pow(x[1], 2)) * (4.0*pow(x[1], 2) + 2.0)) + alfa*(exp(pow(x[0], 2)) + exp(pow(x[1], 2)))', 'u3', \
    #                     '$e^{x^{2}}+e^{y^{2}}$')

    heat_conduction_problem('(pow(x[0], 2) + pow(x[1], 2)) * pow(t, 2)', '(2.0*pow(x[0], 2)/R + 2.0*pow(x[1], 2)/R) * pow(t, 2)', \
                            '2.0*(pow(x[0], 2) + pow(x[1], 2))*t - 4.0*alfa*pow(t, 2)', 'u1', '$(x^{2}+y^{2})t^{2}$')
    
    heat_conduction_problem('sin(x[0]*t) + cos(x[1]*t)', 't*cos(x[0]*t)*x[0]/R - t*sin(x[1]*t)*x[1]/R', \
                            'x[0]*cos(x[0]*t) - x[1]*sin(x[1]*t) + alfa*pow(t, 2)*(sin(x[0]*t) + cos(x[1]*t))', 'u2', '$sin(xt)+cos(yt)$')
    
    heat_conduction_problem('exp(pow(x[0], 2)-t) + exp(pow(x[1], 2)-t)', 'exp(pow(x[0], 2)-t)*2.0*pow(x[0], 2)/R + exp(pow(x[1], 2)-t)*2.0*pow(x[1], 2)/R', \
                        '-(exp(pow(x[0], 2)-t) + exp(pow(x[1], 2)-t)) - alfa*(exp(pow(x[0], 2)-t) * (4.0*pow(x[0], 2) + 2.0) + exp(pow(x[1], 2)-t) * (4.0*pow(x[1], 2) + 2.0))', 'u3', \
                        '$e^{x^{2}-t}+e^{y^{2}-t}$')

    

if __name__ == '__main__':
    main()