import numpy as np
import matplotlib.pyplot as plt
import time as time

# variables:
mu = 0.1
Lx = 1.0
Ly = 1.0
nx = 40
ny = 40
nghost = 1
CFL=0.9

size_x = nx + 2 * nghost
size_y = ny + 2 * nghost

dx = Lx / nx
dy = Ly / ny
nbvar = 3
iu, iv, iP = 0, 1, 2

#Define ranges
inner_range_x = range(nghost, size_x-1)
inner_range_y = range(nghost, size_y-1)
outer_range_x = range(size_x)
outer_range_y = range(size_y)

def compute_time_step():
    vmax=1.0
    dt_conv = min(dx/1.0, dy/1.0)
    dt_diff = (dx**2 * dy**2)/(2*mu*(dx**2+dy**2))
    return CFL*min(dt_diff,dt_conv)

def init(tab):
    # Initializing at rest
    tab[:, :, iP] = 1.0
    tab[:, :, iv] = 0.0
    tab[:, :, iu] = 0.0

def momentum_step(tab, tab_, dt):
    # Evaluate local velocities
    u = tab_[:, :, iu]
    v = tab_[:, :, iv]

    # Upwind choice
    dudx = np.where(u > 0, (u - np.roll(u, 1, axis=0)) / dx, (np.roll(u, -1, axis=0) - u) / dx)
    dvdx = np.where(u > 0, (v - np.roll(v, 1, axis=0)) / dx, (np.roll(v, -1, axis=0) - v) / dx)
    dvdy = np.where(v > 0, (v - np.roll(v, 1, axis=1)) / dy, (np.roll(v, -1, axis=1) - v) / dy)
    dudy = np.where(v > 0, (u - np.roll(u, 1, axis=1)) / dy, (np.roll(u, -1, axis=1) - u) / dy)

    # Evaluate the Laplace operator
    Laplace_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2 * u) / (dx**2) + \
                (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2 * u) / (dy**2)
    Laplace_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2 * v) / (dx**2) + \
                (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2 * v) / (dy**2)

    # Update tab
    tab[:, :, iu] = tab_[:, :, iu] + dt * (-u * dudx - v * dudy + mu * Laplace_u)
    tab[:, :, iv] = tab_[:, :, iv] + dt * (-u * dvdx - v * dvdy + mu * Laplace_v)

    apply_BC_u_only(tab)

def roll(tab, n, axis):
    return np.roll(tab, n, axis)[nghost:size_x-1, nghost:size_y-1]

def poisson_solver(tab, dt, tol=1e-6, nmax=10000):

    #Compute div of uT
    div = np.zeros(np.shape(tab[:,:,iu]))

    for i in inner_range_x:
        for j in inner_range_y:
            div[i,j] = ((tab[i+1,j,iu]-tab[i-1,j,iu])/dx + (tab[i,j+1,iv]-tab[i,j-1,iv])/dy)*(0.5/dt)
    
    # # Evaluate local velocities
    # u = tab[:, :, iu]
    # v = tab[:, :, iv]

    # div = (
    #     roll(u, 1, axis=0)-roll(u, -1, axis=0)/dx 
    # +   roll(v, 1, axis=1)-roll(v, -1, axis=1)/dy
    # )*(0.5/dt)

    err=1000
    niter=0
    C=1.0/(-2*(1.0/dx**2 + 1.0/dy**2))

    P =tab[:,:,iP].copy() #+1
    P_=tab[:,:,iP].copy() #+1

    while (err>tol and niter<nmax):
        #for i in inner_range_x:
        #    for j in inner_range_y:
        #        P[i,j] = (div[i,j] - (P_[i+1,j]+P_[i-1,j])/dx**2 - (P_[i,j+1]+P_[i,j-1])/dy**2)*C      

        P[1:-1, 1:-1] = (div[1:-1, 1:-1]- (P_[2:, 1:-1] + P_[:-2, 1:-1]) / dx**2 - (P_[1:-1, 2:] + P_[1:-1, :-2]) / dy**2) * C

        err = np.max(np.abs(P-P_))
        P,P_=P_,P
        niter+=1

    print(niter,err)

    tab[:,:, iP] = P[:,:].copy()

    apply_BC_P_only(tab)
    return niter
    
def projection_step(tab, dt):
    for i in inner_range_x:
        for j in inner_range_y:
            tab[i, j, iu] = tab[i, j, iu] - dt*(tab[i+1, j, iP]-tab[i-1, j, iP])/(2*dx)
            tab[i, j, iv] = tab[i, j, iv] - dt*(tab[i, j+1, iP]-tab[i, j-1, iP])/(2*dy)
    apply_BC_u_only(tab)


   
def apply_BC(tab):
    #Fill BC
    apply_BC_P_only(tab)
    apply_BC_u_only(tab)

def apply_BC_u_only(tab):
    #Fill BC
    for i in outer_range_x:
        tab[i, 0, iu] = 1.0
        tab[i, 0, iv] = 0.0 

        tab[i, -1, iu] = -1.0
        tab[i, -1, iv] = 0 

    for j in outer_range_y:
        tab[0, j, iu] = 0
        tab[0, j, iv] = 0 

        tab[-1, j, iu] = 0
        tab[-1, j, iv] = 0 

def apply_BC_P_only(tab):
    #Fill BC
    for i in outer_range_x:
        tab[i, 0, iP] = 1.0
        tab[i, -1, iP] = 1.0

    for j in outer_range_y:
        tab[0, j, iP] = tab[1, j, iP]
        tab[-1, j, iP] = tab[-2, j, iP]


# Initialize solution array
sol = np.zeros((size_x, size_y, nbvar))
sol_= np.zeros((size_x, size_y, nbvar))

init(sol_)
apply_BC(sol_)

t=0
dt = compute_time_step()
nstepsmax=100000
nstep=0
tmax=50
done=False
conv=False

time_mom=0
time_poisson=0
time_proj=0

begin=time.time()
while (t<tmax and nstep <= nstepsmax and not done):

    if (t+dt>tmax):
        dt=t-tmax
        done=True
    
    t0=time.time()
    momentum_step(sol, sol_, dt)
    t1=time.time()
    time_mom+=t1-t0
    
    t0=time.time()
    niter = poisson_solver(sol, dt)
    t1=time.time()
    time_poisson+=t1-t0

    t0=time.time()
    projection_step(sol, dt)
    t1=time.time()
    time_proj+=t1-t0

    sol_[:,:,:] = sol[:,:,:].copy()

    t+=dt
    nstep+=1

    if (niter==1 and not conv):
        nstep=nstepsmax-500
        conv=True
        print("Stopping in 500 steps as we have converged it seems")

end=time.time()

total_time=end-begin
print("total time=", total_time)
print("mom:", time_mom*100/total_time, "%", time_mom, "s")
print("poisson:", time_poisson*100/total_time, "%", time_poisson, "s")
print("proj:", time_proj*100/total_time, "%", time_proj, "s")

x = np.linspace(0.0, Lx, size_x)
y = np.linspace(0.0, Ly, size_y)

X, Y = np.meshgrid(x, y)

plt.style.use("dark_background")
plt.figure()
plt.contourf(X[:, :], Y[:, :], sol[:, :, iP], cmap="coolwarm")
plt.colorbar()

plt.quiver(X[:, :], Y[:, :], sol[:, :, iv], sol[:, :, iu], color="black")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.savefig("out.png")
