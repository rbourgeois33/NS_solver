import numpy as np
import matplotlib.pyplot as plt
import time as time

# variables:
mu = 0.1
Lx = 1.0
Ly = 1.0
nx = 100
ny = 100
nghost = 1
CFL=0.99

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
    for i in inner_range_x:
        for j in inner_range_y:
            tab[i, j, iP] = 0.0
            tab[i, j, iv] = 0.0
            tab[i, j, iu] = 0.0

# def momentum_step(tab, tab_):
#     #Apply momentum step
#     for i in inner_range_x:
#         for j in inner_range_y:

#             #Evaluate local velocities
#             u = tab_[i, j, iu]
#             v = tab_[i, j, iv]

#             #Upwind choice
#             dudx = (u - tab_[i-1, j, iu])/dx if u>0 else (tab_[i+1, j, iu] - u)/dx
#             dvdx = (v - tab_[i-1, j, iv])/dx if u>0 else (tab_[i+1, j, iv] - v)/dx
#             dvdy = (v - tab_[i, j-1, iv])/dy if v>0 else (tab_[i, j+1, iv] - v)/dy
#             dudy = (u - tab_[i, j-1, iu])/dy if v>0 else (tab_[i, j+1, iu] - u)/dy

#             #Evaluate the Laplace operator

#             Laplace_u = (tab_[i+1, j, iu] + tab_[i-1, j, iu] - 2*u)/(dx**2) + (tab_[i, j+1, iu] + tab_[i, j-1, iu] - 2*u)/(dy**2)
#             Laplace_v = (tab_[i+1, j, iv] + tab_[i-1, j, iv] - 2*v)/(dx**2) + (tab_[i, j+1, iv] + tab_[i, j-1, iv] - 2*v)/(dy**2)

#             #Update tab
#             tab[i, j, iu] = tab_[i, j, iu] + dt*(-u*dudx - v*dudy + mu * Laplace_u)
#             tab[i, j, iv] = tab_[i, j, iv] + dt*(-u*dvdx - v*dvdy + mu * Laplace_v)

def momentum_step(tab, tab_):
    # Extract the required fields for better readability and performance
    u = tab_[:, :, iu]
    v = tab_[:, :, iv]
    
    # Compute upwind derivatives using vectorized operations
    dudx = np.where(u > 0, (u - np.roll(u, 1, axis=0)) / dx, (np.roll(u, -1, axis=0) - u) / dx)
    dudy = np.where(v > 0, (u - np.roll(u, 1, axis=1)) / dy, (np.roll(u, -1, axis=1) - u) / dy)
    
    dvdx = np.where(u > 0, (v - np.roll(v, 1, axis=0)) / dx, (np.roll(v, -1, axis=0) - v) / dx)
    dvdy = np.where(v > 0, (v - np.roll(v, 1, axis=1)) / dy, (np.roll(v, -1, axis=1) - v) / dy)
    
    # Compute Laplace operators using finite difference approximations
    Laplace_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2*u) / dx**2 + \
                (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2*u) / dy**2
                
    Laplace_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2*v) / dx**2 + \
                (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2*v) / dy**2
    
    # Update tab with momentum equation (vectorized)
    tab[:, :, iu] = tab_[:, :, iu] + dt * (-u * dudx - v * dudy + mu * Laplace_u)
    tab[:, :, iv] = tab_[:, :, iv] + dt * (-u * dvdx - v * dvdy + mu * Laplace_v)


def poisson_solver(tab, tol=1e-10, nmax=1000):

    #Compute div of uT
    div = np.zeros(np.shape(tab[:,:,iu]))

    # for i in inner_range_x:
    #     for j in inner_range_y:
    #         div[i,j] = ((tab[i+1,j,iu]-tab[i-1,j,iu])/dx + (tab[i,j+1,iv]-tab[i,j-1,iv])/dy)*(0.5/dt)

    div[1:-1, 1:-1] = ((tab[2:, 1:-1, iu] - tab[:-2, 1:-1, iu]) / (2 * dx) + 
                       (tab[1:-1, 2:, iv] - tab[1:-1, :-2, iv]) / (2 * dy)) * (1.0 / dt)

    
    err=1000
    niter=0
    C=1.0/(-2*(1.0/dx**2 + 1.0/dy**2))

    P =tab[:,:,iP] #+1
    P_=tab[:,:,iP] #+1

    while (err>tol and niter<nmax):
        # for i in inner_range_x:
        #     for j in inner_range_y:
        #         P[i,j] = (div[i,j] - (P_[i+1,j]+P_[i-1,j])/dx**2 - (P_[i,j+1]+P_[i,j-1])/dy**2)*C                
        P[1:-1, 1:-1] = (div[1:-1, 1:-1] - (P_[2:, 1:-1] + P_[:-2, 1:-1]) / dx**2 - (P_[1:-1, 2:] + P_[1:-1, :-2]) / dy**2) * C
         
        err = np.max(np.abs(P-P_))
        #P_=np.copy(P)
        P,P_=P_,P
        niter+=1

    tab[:, :, iP] = P
    

def projection_step(tab):
  
#   for i in inner_range_x:
#             for j in inner_range_y:
#                 tab[i, j, iu] = tab[i, j, iu] - dt*(tab[i+1, j, iP]-tab[i-1, j, iP])/(2*dx)
#                 tab[i, j, iv] = tab[i, j, iv] - dt*(tab[i, j+1, iP]-tab[i, j-1, iP])/(2*dy)
    # Update u-component of velocity
    tab[1:-1, 1:-1, iu] -= dt * (tab[2:, 1:-1, iP] - tab[:-2, 1:-1, iP]) / (2 * dx)

    # Update v-component of velocity
    tab[1:-1, 1:-1, iv] -= dt * (tab[1:-1, 2:, iP] - tab[1:-1, :-2, iP]) / (2 * dy)



def apply_BC(tab):
    #Fill BC
    for i in outer_range_x:
        tab[i, 0, iu] = 1.0
        tab[i, 0, iv] = 0.0
        tab[i, 0, iP] = 0.0

        tab[i, -1, iu] = 0
        tab[i, -1, iv] = 0 
        tab[i, -1, iP] = tab[i, -2, iP]

    for j in outer_range_y:
        tab[0, j, iu] = 0
        tab[0, j, iv] = 0 
        tab[0, j, iP] = tab[1, j, iP]

        tab[-1, j, iu] = 0
        tab[-1, j, iv] = 0 
        tab[-1, j, iP] = tab[-2, j, iP]

def apply_BC_u_only(tab):
    #Fill BC
    for i in outer_range_x:
        tab[i, 0, iu] = 1.0
        tab[i, 0, iv] = 0.0 

        tab[i, -1, iu] = 0
        tab[i, -1, iv] = 0 

    for j in outer_range_y:
        tab[0, j, iu] = 0
        tab[0, j, iv] = 0 

        tab[-1, j, iu] = 0
        tab[-1, j, iv] = 0 




# Initialize solution array
sol = np.zeros((size_x, size_y, nbvar))
sol_= np.zeros((size_x, size_y, nbvar))

init(sol_)
apply_BC(sol_)

t=0
dt = compute_time_step()
nstepsmax=100
nstep=0
tmax=0.1
done=False

time_mom=0
time_poisson=0
time_proj=0

begin=time.time()
while (t<tmax and nstep <= nstepsmax and not done):

    if (t+dt>tmax):
        dt=t-tmax
        done=True
    
    t0=time.time()
    momentum_step(sol, sol_)
    t1=time.time()
    time_mom+=t1-t0
    
    apply_BC_u_only(sol)

    t0=time.time()
    poisson_solver(sol)
    t1=time.time()
    time_poisson+=t1-t0

    t0=time.time()
    projection_step(sol)
    t1=time.time()
    time_proj+=t1-t0

    
    apply_BC(sol)

    sol_[:,:,:] = sol[:,:,:]

    t+=dt
    nstep+=1
    #print(t, dt)
end=time.time()

total_time=end-begin
print("total time=", total_time)
print("mom:", time_mom*100/total_time, "%", time_mom, "s")
print("poisson:", time_poisson*100/total_time, "%", time_poisson, "s")
print("proj:", time_proj*100/total_time, "%", time_proj, "s")

x = np.linspace(0.0, Lx, nx+2*nghost)
y = np.linspace(0.0, Ly, ny+2*nghost)

X, Y = np.meshgrid(x, y)

# The [::2, ::2] selects only every second entry (less cluttering plot)
plt.style.use("dark_background")
plt.figure()
plt.contourf(X[::2, ::2], Y[::2, ::2], sol[::2, ::2, iP], cmap="coolwarm")
plt.colorbar()

plt.quiver(X[::2, ::2], Y[::2, ::2], sol[::2, ::2, iv], sol[::2, ::2, iu], color="black")
# plt.streamplot(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()
