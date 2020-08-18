import time
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from random import randint
import pandas as pd
from numba import njit, stencil, jit
colors = ['r', 'b', 'm']

def interweave(arr1, arr2):
    """
    This is an auxiliarry function that is used in the creation of the diagonals. It takes two arrays and interweave
    their values.
    
    Arguments:
    arr1 -- array 1
    arr2 -- array 2
    
    Return:
    an array with values from arr1 and arr2 interweaved. 
    """
    res = np.empty(arr1.size + arr2.size, dtype=arr2.dtype)
    res[0::2] = arr1.flatten()
    res[1::2] = arr2.flatten()
    return res



#Variables:

#Fluid properties:
##viscositites functions
visc_o = lambda p: 2.5*p/p #cp
visc_g = lambda p: (3e-10)*p**2 + (1e-6)*p + 0.0133 #cp
dvisc_g = lambda p: 6e-10*p + 1e-6
Co = 0.8e-5 #psi-1

#FVF, relative permeabilities, oil_gas ratio functions
b_o = lambda p, Pbub, Patm: (np.exp(8e-5*(Patm - p))*(p < Pbub) 
                                            + np.exp(8e-5*(Patm - Pbub))*np.exp(Co*(p - Pbub))*(p >= Pbub))
b_g = lambda p, Pbub, Patm: np.exp(-1.7e-3*(Patm - p))*(p < Pbub) + np.exp(-1.7e-3*(Patm - Pbub))*(p >= Pbub)
R_s  = lambda p, Pbub: (178.11**2/5.615)*(p/Pbub)**(1.3)*(p < Pbub) + (178.11**2/5.615)*(p >= Pbub)
k_ro = lambda Sg: (1-Sg)**(1.5)
k_rg = lambda Sg: Sg**2

#derivatives
db_o = lambda p, Pbub, Patm: (-8e-5*np.exp(8e-5*(Patm - p))*(p < Pbub) 
                                            + (Co)*np.exp(8e-5*(Patm - Pbub))*np.exp(Co*(p - Pbub))*(p >= Pbub))
db_g = lambda p, Pbub, Patm: 1.7e-3*np.exp(-1.7e-3*(Patm - p))*(p < Pbub)
dR_s = lambda p, Pbub: (178.11**2/5.615)*1.3/Pbub*(p/Pbub)**(0.3)*(p < Pbub)
dk_ro = lambda Sg: -1.5*(1-Sg)**(0.5)
dk_rg = lambda Sg: 2*Sg

#geometric transmissibility computation.
T = lambda K,i,j,dx,dy,dz: (dx[i] + dx[j])*(dx[i]/K[i] + dx[j]/K[j])**(-1)*dy[i]*dz[i]/dx[i]

#transmissiblity's fluid part upwinding function
def H(k_r,visc,b):   
    return  k_r*b/visc                 

#Rs*Transmissiblity's fluid part upwinding function
HRs = lambda Rs,P,k_r,visc,b,i,j: Rs[i]*k_r[i]*b[i]/visc[i] if (P[i] > P[j]) \
                                       else Rs[j]*k_r[j]*b[j]/visc[j]

#derivatives of transmissiblity's fluid part (oil and gas) in repsect to pressure and saturation.
def dHop(k_r,visc,db):
    return k_r*db/visc

def dHs(dk_r,visc,b):
    return dk_r*b/visc

def dHgp(k_r,visc,dvisc,b,db):
    return k_r/visc*(db - dvisc*b/visc)

def dHRs_p(dRs,Rs,k_r,visc,db,b):
    return k_r/visc*(Rs*db + dRs*b)

def dHRs_s(Rs,dk_r,visc,b):
    return Rs*b*dk_r/visc

#Well treatment functions
Tw = lambda WI,k_r,visc,b: WI*k_r*b/visc
dTop = lambda WI,k_r,visc,db: WI*k_r*db/visc
dTgp = lambda WI,k_r,visc,dvisc,b,db: WI*k_r/visc*(db-dvisc*b/visc)
dTgop = lambda WI,dRs,Rs,P,k_r,visc,db,b: WI*k_r/visc*(Rs*db + dRs*b)
                                                   
dTs = lambda WI,dk_r,visc,b: WI*dk_r*b/visc



#Objects definition:
"""We are going to define all the classes of objects we need for this project: Rock, Fluid, Grid and Wells."""

class Rock:
    def __init__(self, compressibility, permeability_X, permeability_Y, permeability_Z, porosity):
        self.Cr = compressibility     #psi^-1
        self.Kx = permeability_X      #md
        self.Ky = permeability_Y
        self.Kz = permeability_Z
        self.phi = porosity
        
class Fluids:
    def __init__(self, viscosities, densities, compressibilities):
        self.mu = viscosities           #cP
        self.rho_f = densities          #lbm/ft^3
        self.Cf = compressibilities     #psi^-1
        

class Grid:
    #grid_dims = [Nx, Ny, Nz] number of grids in the three directions
    def __init__(self, Length_X, Length_Y, Length_Z, rock, grid_dims, grid_discrt):
        self.Lx = Length_X            #ft
        self.Ly = Length_Y
        self.Lz = Length_Z
        self.space_steps = {"dX": np.array(grid_discrt["x"]),          #delta X, Y, Z
                            "dY": np.array(grid_discrt["y"]), 
                            "dZ": np.array(grid_discrt["z"])}                   
        self.cell_vol = self.space_steps["dX"]*self.space_steps["dY"]*self.space_steps["dZ"]  # cell volume ft^3
        
class Wells:
    def __init__(self, grid_dims, alpha):  
        self.Nx = grid_dims["x"]
        self.Ny = grid_dims["y"]
        self.Nz = grid_dims["z"]
        self.alpha = alpha
        self.wells = np.zeros((grid_dims["x"]*grid_dims["y"], 4))
        self.wells_pos = {}
    
    def add_well(self, pos, rate, wellName, wellBHP, welldiam, wellControl, kx, ky, dx, dy, dz):
        """
        Add a single well to the wells vector.
        
        Argument:
        position_X -- the X position of the well.
        position_Y -- the Y position of the well
        rate -- the injection or production rate of the well
        wellName -- the wellName: producer or injector
        """
        ro = 0.28*np.sqrt(((ky/kx)**0.5)*dx**2 + ((kx/ky)**0.5)*dy**2)/((ky/kx)**0.25 + (kx/ky)**0.25)
        k  = (kx*ky)**0.5
        
        self.wells_pos[wellName] = pos
        self.wells[pos,0] = rate if wellName[0] == 'P' else -rate
        self.wells[pos,1] = wellBHP
        self.wells[pos,2] = self.alpha*2*np.pi*dz*k/np.log(2*ro/welldiam)
        self.wells[pos,3] = wellControl=='BHP'
     
    def add_wells(self, positions_X, positions_Y, rates, wellNames, wellBHPs , wellDiams, wellControl, rock, grid):
        """
        Add a multiple wells to the wells vector.
        
        Argument:
        positions_X -- an array of the X positions of the well.
        positions_Y -- an array of the Y positions of the well.
        rates -- an array of the injection or production rates of the wells    #STB/day
        wellNames -- an array of the wells type: producer or injector
        """
        Kx, Ky = rock.Kx, rock.Ky
        dX, dY, dZ = grid.space_steps["dX"], grid.space_steps["dY"], grid.space_steps["dZ"]
        for i in range(len(positions_X)):
            pos = (positions_Y[i] - 1)*self.Nx + positions_X[i] - 1
            kx, ky = Kx[pos], Ky[pos]
            dx, dy, dz = dX[pos], dY[pos], dZ[pos]
            self.add_well(pos, rates[i], wellNames[i], wellBHPs[i], wellDiams[i],\
                             wellControl[i], kx, ky, dx, dy, dz)
            
class Interfaces:
    """
    This the class representing interfaces. It has different memeber variables: grid dimensions,
    geometric transmissibility, functions (which contains all the functions and their derivatives)
    variables (all the variables updated with the last pressure vector), transmissibility matrix and
    compressibility matrix which are going to be used to compute the residual and the jacobian in a
    separate class (Jacobian).
    """
    
    def __init__(self, grid_dims, rock, grid, alpha, functions, var):
        """
        Initializing an object of this class.
        
        Arguments:
        grid_dims -- a dict containing the grid dimensions.
        rock      -- the rock object containing the permeabilities.
        grid      -- the grid object containing the discretization.
        alpha     -- the conversion factor.
        functions -- a dict containing all the needed functions.
        """
        self.Nx = grid_dims["x"]
        self.Ny = grid_dims["y"]
        self.Nz = grid_dims["z"]
        nx, ng = self.Nx, self.Nx*self.Ny
        Kx, Ky, T = rock.Kx, rock.Ky, functions['T']
        dx, dy, dz = grid.space_steps["dX"], grid.space_steps["dY"], grid.space_steps["dZ"]
        self.transm_geom = {"x": alpha*np.array([T(Kx,i,i+1,dx,dy,dz) if (i+1)%nx else 0 for i in range(ng-1)]),
                           "y": alpha*np.array([T(Ky,i,i+nx,dy,dx,dz) if (i+nx) < ng else 0 for i in range(ng-nx)]),
                           "z": 0}
                           #geometric transmissibilities
        #Creating the connections list for this type of geometry             
        self.connectX = np.zeros((ng - 1, 2), dtype = np.int32)
        self.connectY = np.zeros((ng - nx, 2), dtype = np.int32)
        for i in range(ng - 1):
            if (i+1)%nx:
                self.connectX[i] = [i, i + 1]
            if (i+nx) < ng and i < ng - nx:
                self.connectY[i] = [i, i + nx]
        self.functions = functions
        self.vars = var
        self.transmissibilities = None  #sparse matrix for transmissibilities
        self.compressibilities  = None
     
    def diag(self, H, direction, size):
        """
        Create a diagonal. And because we are doing this separately for oil
        and gas equations we should interweave with zeros. Therefore, the diagonal obtained from oil, for example,
        can be directly added to the one obtained from gas (for the same offset position).
        
        Arguments:
        H         -- the upwinded fluid term for a diagonal position.
        direction -- x or y, to choose the right geometrci transmissibility
        size      -- the number of interfaces, correspond to the one in the x direction of y direction
        
        Return:
        A diagonal in the transmissibility matrix.
        """
        res = np.zeros(2*size, dtype=np.float64)
        res[::2] = self.transm_geom[direction]*H
        return res
        
    def update_variables(self, P, Pn):
        """
        Updating the variables using the last pressure vector.
        
        Arguments:
        P -- the last pressure vector, containing the cell saturations too.
        Pn -- the pressure at the previous time step
        
        """
        v = self.vars                    #the variables going to be updated
        f = self.functions               #the functions used to update the variables
        p, s = P[::2], P[1::2]           #pressure and saturation
        Pb, Pa = v["Pbub"], v["Patm"]
        
        #updating the parameters in the variables member using the functions.
        v['visco'], v['viscg'] = f['visco'](p), f['viscg'](p)
        v['bo'], v['bg'], v['Rs'] = f['bo'](p, Pb, Pa), f['bg'](p, Pb, Pa), f['Rs'](p, Pb)
        v['ko'], v['kg'] = f['ko'](s), f['kg'](s)
        
        v['bo_n'], v['bg_n'], v['Rs_n'] = f['bo'](Pn[::2], Pb, Pa), f['bg'](Pn[::2], Pb, Pa), f['Rs'](Pn[::2], Pb)
        
        #updating the derivatives.
        v['dviscg'] = f['dviscg'](p)
        v['dbo'], v['dbg'], v['dRs'] = f['dbo'](p, Pb, Pa), f['dbg'](p, Pb, Pa), f['dRs'](p, Pb)
        v['dko'], v['dkg'] = f['dko'](s), f['dkg'](s)
        
    def update_compressibilities(self, grid, rock, fluid, dt):
        """
        Creating the compressibility matrix.
        
        Argument:
        grid -- the grid object containing cell volume and grid dimensions
        rock -- the rock object containing porosity
        fluid -- the fluid object to get fluid compressibility
        dt -- is the time step
        """
        self.compressibilities  = diags(np.repeat(grid.cell_vol*rock.phi/(5.615*dt), 2), 0,\
                                                            shape=(2*self.Nx*self.Ny, 2*self.Nx*self.Ny))
        return self.compressibilities
    
    def update_transmissibilities(self, P):
        """
        Creating the transmissibilities matrix by forming the diagonals first, there are 2 main diagonals one for oil
        and one for gas, and there is 4 other secondary diagonals on each side. We construct only the diagonlas on 
        one side and we use the symmetrie to deduce the others.
        
        Argument:
        P -- Pressure vector.
        
        Return:
        Transmissibility matrix that is going to be used in forming the Jacobian and the Residual.
        """
        nx, ng = self.Nx, self.Nx*self.Ny           #grid dimensions
        #Upwinding
        Hox, Hoy, HRs_ox, HRs_oy, Hgx, Hgy = self.upwinding(P)   #Upwinded fluid part of transmissibilitites
        #construct the diags:
        diag1_o, diagNx_o, diag_o, diag_g, diag1_g, diagNx_g = \
                        self.construct_diag(Hox, Hoy, HRs_ox, HRs_oy, Hgx, Hgy)
        #constructing the transmissibility matrix from the diagonals
        self.transmissibilities = diags([diag1_o, diag1_o[1:-1], diag_o, diagNx_o, diagNx_o[1:-1], \
                                             diag1_g, diag1_g, diag_g, diagNx_g, diagNx_g], \
                                                    [1, -3, -1, 2*nx-1, -2*nx-1, 2, -2, 0, -2*nx, 2*nx],\
                                                            shape=(2*ng, 2*ng)) 
        return self.transmissibilities
    
    def upwinding(self, P):
        """
        Upwinding the fluid part of transmissibilities.
        
        Argument:
        P -- the pressure vector
        
        Return:
        The constructed upwinded fluid part of transmissibilities for all directions 
        """
        nx, ny, v, f = self.Nx, self.Ny, self.vars, self.functions
        visco, viscg = v['visco'], v['viscg']
        bo, bg, Rs, ko, kg = v['bo'], v['bg'], v['Rs'], v['ko'], v['kg']
        
        H = f['H']
        #upwinding oil transmissibilities 
        backwardX = P[::2][self.connectX[:,0]] >= P[::2][self.connectX[:,1]]
        backwardY = P[::2][self.connectY[:,0]] >= P[::2][self.connectY[:,1]]
        
        Hox = H(ko[:-1],visco[:-1],bo[:-1])*backwardX + H(ko[1:],visco[1:],bo[1:])*(1-backwardX)
        Hox[nx-1::nx] = np.zeros((ny-1,1)).flatten()
        
        Hoy = H(ko[:-nx],visco[:-nx],bo[:-nx])*backwardY + H(ko[nx:],visco[nx:],bo[nx:])*(1-backwardY)
        #oil transmissibilities mutiplied with Rs
        HRs_ox = Rs[:-1]*H(ko[:-1],visco[:-1],bo[:-1])*backwardX + Rs[1:]*H(ko[1:],visco[1:],bo[1:])*(1-backwardX)
        HRs_ox[nx-1::nx] = np.zeros((ny-1,1)).flatten()
        
        HRs_oy = Rs[:-nx]*H(ko[:-nx],visco[:-nx],bo[:-nx])*backwardY + Rs[nx:]*H(ko[nx:],visco[nx:],bo[nx:])*(1-backwardY)
        #upwinding gas transmissibilities
        Hgx = H(kg[:-1],viscg[:-1],bg[:-1])*backwardX + H(kg[1:],viscg[1:],bg[1:])*(1-backwardX)
        Hgx[nx-1::nx] = np.zeros((ny-1,1)).flatten()
        
        Hgy = H(kg[:-nx],viscg[:-nx],bg[:-nx])*backwardY + H(kg[nx:],viscg[nx:],bg[nx:])*(1-backwardY)

        
        return Hox, Hoy, HRs_ox, HRs_oy, Hgx, Hgy         
    
    def construct_diag(self, Hox, Hoy, HRs_ox, HRs_oy, Hgx, Hgy):
        """
        Constructing the diagonals of the transmissibility matrix from the upwinded tranmissibilities.
        
        Arguments:
        Hox -- the upwinded fluid part of oil transmissibilities in the x direction
        Hoy -- the upwinded fluid part of oil transmissibilities in the y direction
        HRs_ox -- the upwinded fluid part of oil transmissibilities * Rs in the x direction
        HRs_oy -- the upwinded fluid part of oil transmissibilities * Rs in the y direction
        Hgx -- the upwinded fluid part of gas transmissibilities in the x direction
        Hgy -- the upwinded fluid part of gas transmissibilities in the y direction
        
        Return:
        The set of diagonals needed for the transmissibility matrix.
        """
        nx, ng = self.Nx, self.Nx*self.Ny
        #constructing the offset diagonal
        diag1_g = self.diag(Hgx + HRs_ox, "x", ng-1)
        diag1_o = self.diag(Hox, "x", ng-1)
        diagNx_g = self.diag(Hgy + HRs_oy, "y", ng-nx)
        diagNx_o = self.diag(Hoy, "y", ng-nx)
        #constructing the main diagonals   
        diag_g = -np.pad(diagNx_g,(2*nx, 0), 'constant')
        diag_g[:-2*nx] -= diagNx_g
        diag_g[:-2] -= diag1_g
        diag_g[2:] -= diag1_g
        
        diag_o = -np.pad(diagNx_o[:-1],(2*nx, 0), 'constant')
        diag_o[:-2*nx +1] -= diagNx_o
        diag_o[:-1] -= diag1_o
        diag_o[2:] -= diag1_o[:-1]

        diag1_o = np.pad(diag1_o,(1, 0), 'constant')
        diagNx_o= np.pad(diagNx_o,(1, 0), 'constant')
        
        return (diag1_o, diagNx_o, diag_o, diag_g, diag1_g, diagNx_g)

    
class Jacobian:
    """
    The class responsible for computing the residual and the jacobian.
    """
    def __init__(self, interfaces, pressure, Pn, wells):
        """
        Initializing an object of the class using the interfaces, current pressure vector and pressure vector
        at time step n.
        """
        self.Nx, self.Ny, self.Nz = interfaces.Nx, interfaces.Ny, interfaces.Nz
        self.functions = interfaces.functions              #recovering the functions from interfaces
        self.vars = interfaces.vars                        #recovering all the updated variables
        self.trans_geom = interfaces.transm_geom           #recovering the geometric transmissibilities

        self.A = self.accumulation(pressure, Pn)     #the accumulation vector for the jacobian 
        self.T = interfaces.transmissibilities   
        self.C = interfaces.compressibilities
        self.diagWell  = self.well_src(wells, pressure, self.Nx, self.Ny, self.Nz)

        self.residual  = self.T@pressure - self.C@self.A - self.diagWell[0]    #Residual vector computation
        #Well transmissibilities. They will be needed when updating the pressure in the wells and the flow rates
        self.wellTrans = self.diagWell[4:]
        self.jacobian = None  #Jacobian matrix
        #the diagonals tat going to be used in the Jacobian construction
        self.diags = self.T.diagonal(1), self.T.diagonal(2*self.Nx-1),\
                            self.T.diagonal(2), self.T.diagonal(2*self.Nx)
        self.connectX = interfaces.connectX
        self.connectY = interfaces.connectY
        
    def well_src(self, w, p, nx, ny, nz):
        nx, ny, ng, v, f = self.Nx, self.Ny, self.Nx*self.Ny, self.vars, self.functions
        res = csr_matrix((2*nx*ny, 1)) 
        
        W = w.wells
        WI = W[:,2]
        rates = W[:,0]
        Pw = W[:,1]
        BHPControl = W[:,3]
        P = p[::2]
        
        visco, viscg, dviscg = v['visco'], v['viscg'], v['dviscg']
        bo, bg, ko, kg, Rs = v['bo'], v['bg'], v['ko'], v['kg'], v['Rs']
        dbo, dbg, dko, dkg, dRs = v['dbo'], v['dbg'], v['dko'], v['dkg'], v['dRs']
        
        Tw, dTo_p, dTg_p, dT_s, dTgo_p = f['Tw'], f['dTop'], f['dTgp'], f['dTs'], f['dTgop']
        
        To, Tgg = Tw(WI,ko,visco,bo), Tw(WI,kg,viscg,bg)
        Tgo = Rs*To
        dTop, dTggp  = dTo_p(WI,ko,visco,dbo), dTg_p(WI,kg,viscg,dviscg,bg,dbg)
        dTos, dTggs  = dT_s(WI,dko,visco,bo), dT_s(WI,dkg,viscg,bg)
        dTgop, dTgos = dTgo_p(WI,dRs,Rs,P,ko,visco,dbo,bo), Rs*dTos
        
        #contribution to the residual:
        #To avoid division by zero:
        To_1 = np.ones(To.shape)
        for wellname, pos in w.wells_pos.items():
            To_1[pos] = 1/To[pos]
            
        res = interweave((Tgg + Tgo)*(P - Pw)*BHPControl + (Tgg + Tgo)*To_1*rates*(1 - BHPControl),
                             To*(P - Pw)*BHPControl + rates*(1 - BHPControl))

        #contribution to the Jacobian:
        dQop = (To + dTop*(P - Pw))*BHPControl
        dQos = dTos*(P - Pw)*BHPControl
        
        dQgp = (Tgg + Tgo + (dTggp + dTgop)*(P - Pw))*BHPControl + ((dTggp+dTgop)*To - dTop*(Tgg+Tgo))*rates*(To_1**2)*(1 - BHPControl)
        dQgs = (dTggs + dTgos)*(P - Pw)*BHPControl + ((dTggs+dTgos)*To - dTos*(Tgg + Tgo))*rates*(To_1**2)*(1 - BHPControl)
        #constructing the diagonals that are going to be added to the jacobian
        diag0 = interweave(dQgp, dQos)
        diag1 = interweave(dQgs, np.full((len(dQgs),1), 0.))
        diag_1= interweave(dQop, np.full((len(dQop),1), 0.))
    
        return res, diag_1, diag0, diag1, To_1, Tgo, Tgg
                
    
    def diag(self, H, direction, size):
        """
        Creating a diagonal. And because we are doing this separately for oil
        and gas equations we should interweave with zeros. Therefore, the diagonal obtained from oil, for example,
        can be directly added to the one obtained from gas (for the same offset position).
        
        Arguments:
        H         -- the upwinded fluid term for a diagonal position.
        direction -- x or y, to choose the right geometrci transmissibility
        size      -- the number of interfaces, correspond to the one in the x direction of y direction
        
        Return:
        A diagonal in the Jacobian matrix.
        """
        res = np.zeros(2*size, dtype=np.float64)
        res[::2] = self.trans_geom[direction]*H
        return res
    
    def accumulation(self, P, Pn):
        """
        Computing the accumulation term for the residual computations.
        
        Arguments:
        P -- pressure vector.
        Pn -- pressure vector at time step n.
        
        Return:
        The accumulation vector that will be multiplied by compressibility matrix and added to the residual.
        """
        v = self.vars
        S = P[1::2]
        b_o, R_s, b_g = v["bo"], v["Rs"], v["bg"]
        b_on, Rs_n, b_gn = v["bo_n"], v["Rs_n"], v["bg_n"]
        #gas accumulation
        gas_accum = (b_g*S - b_gn*Pn[1::2]) \
                            + R_s*(1-S)*b_o - Rs_n*(1-Pn[1::2])*b_on

        #oil accumulation
        oil_accum = b_o*(1-S) - b_on*(1-Pn[1::2])
                    
        return interweave(gas_accum, oil_accum) 
            
    def accum_jacobian(self, P):
        """
        Computing the accumulation term for the Jacobian computations.
        
        Arguments:
        P -- pressure vector.
        
        Return:
        The accumulation elements that will be added to the main diagonals in the Jacobian matrix.
        """
        ng, v = self.Nx*self.Ny, self.vars
        P, S = P[::2], P[1::2]
        db_g, dR_s, b_o, Rs, db_o, b_g = v["dbg"], v["dRs"], v["bo"], v["Rs"], v["dbo"], v["bg"] 
        #gas accumulation for the diagonals of the Jacobian
        J_gp_accum = db_g*S + (1-S)*(dR_s*b_o + Rs*db_o)
        J_gs_accum = b_g - Rs*b_o
        
        #oil accumulation for the diagonals of the Jacobian
        J_os_accum = b_o
        J_op_accum = (1-S)*db_o
        
        C = self.C
        return  C@interweave(J_gp_accum, -J_os_accum), \
                            C@interweave(J_gs_accum, np.zeros((ng,1))), \
                                          C@interweave(J_op_accum, np.zeros((ng,1)))

    def derivate(self, P):
        """
        Derivating the fluid part of transmissibilities (oil, oil*Rs, gas) in respect to pressure and saturation.
        It returns 4 vectors because we are doing this forward, in x and y directions, and backward also in x and y.
        The 'op' in the switcher dictionary means the derivative of oil in respect to pressure, for 'sg' is the 
        derivative of gas equation in respect to saturation, and 'Rs_p' is the derivative of Rs*(fluid part of oil
        transmissibility) in repsect to pressure, the same meanings can be deduced for the other definitions.
        
        Arguments:
        P -- pressure vector.
        
        Return:
        A dictionary containing all the derivatives.
        """
        #Retrieving the updated variables and the derivative functions.
        nx, ny, v, f = self.Nx, self.Ny, self.vars, self.functions
        visco, viscg, dviscg = v['visco'], v['viscg'], v['dviscg']
        bo, bg, Rs, ko, kg = v['bo'], v['bg'], v['Rs'], v['ko'], v['kg']
        dbo, dbg, dRs, dko, dkg = v['dbo'], v['dbg'], v['dRs'], v['dko'], v['dkg']
        dHop, dHgp, dHs, dHRs_p = f['dHop'], f['dHgp'], f['dHs'], f['dHRs_p']
        p = P[::2]
        
        backwardX = p[self.connectX[:,0]] >= p[self.connectX[:,1]]
        backwardY = p[self.connectY[:,0]] >= p[self.connectY[:,1]]
        
        #derivative of oil in respect to pressure backward and forward in both directions.
        dHpo_bx, dHpo_fx = dHop(ko[:-1],visco[:-1],dbo[:-1])*backwardX, dHop(ko[1:],visco[1:],dbo[1:])*(1-backwardX)
        dHpo_bx[nx-1::nx], dHpo_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
        
        dHpo_by = dHop(ko[:-nx],visco[:-nx],dbo[:-nx])*backwardY
        dHpo_fy = dHop(ko[nx:],visco[nx:],dbo[nx:])*(1-backwardY)
        #derivative of oil in respect to saturation backward and forward in both directions.
        dHso_bx, dHso_fx = dHs(dko[:-1],visco[:-1],bo[:-1])*backwardX, dHs(dko[1:],visco[1:],bo[1:])*(1-backwardX)
        dHso_bx[nx-1::nx], dHso_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
    
        dHso_by = dHs(dko[:-nx],visco[:-nx],bo[:-nx])*backwardY
        dHso_fy = dHs(dko[nx:],visco[nx:],bo[nx:])*(1-backwardY)
        #derivative of gas in respect to saturation backward and forward in both directions.
        dHsg_bx, dHsg_fx = dHs(dkg[:-1],viscg[:-1],bg[:-1])*backwardX, dHs(dkg[1:],viscg[1:],bg[1:])*(1-backwardX)
        dHsg_bx[nx-1::nx], dHsg_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
    
        dHsg_by = dHs(dkg[:-nx],viscg[:-nx],bg[:-nx])*backwardY
        dHsg_fy = dHs(dkg[nx:],viscg[nx:],bg[nx:])*(1-backwardY)
        #derivative of gas in respect to pressure backward and forward in both directions.
        dHpg_bx, dHpg_fx = dHgp(kg[:-1],viscg[:-1],dviscg[:-1],bg[:-1],dbg[:-1])*backwardX, dHgp(kg[1:],viscg[1:],dviscg[1:],bg[1:],dbg[1:])*(1-backwardX)
        dHpg_bx[nx-1::nx], dHpg_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
    
        dHpg_by = dHgp(kg[:-nx],viscg[:-nx],dviscg[:-nx],bg[:-nx],dbg[:-nx])*backwardY
        dHpg_fy = dHgp(kg[nx:],viscg[nx:],dviscg[nx:],bg[nx:],dbg[nx:])*(1-backwardY)
        #derivative of Rs*oil in respect to pressure backward and forward in both directions.
        dHRp_bx, dHRp_fx = dHRs_p(dRs[:-1],Rs[:-1],ko[:-1],visco[:-1],dbo[:-1],bo[:-1])*backwardX, dHRs_p(dRs[1:],Rs[1:],ko[1:],visco[1:],dbo[1:],bo[1:])*(1-backwardX)
        dHRp_bx[nx-1::nx], dHRp_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
        
        dHRp_by = dHRs_p(dRs[:-nx],Rs[:-nx],ko[:-nx],visco[:-nx],dbo[:-nx],bo[:-nx])*backwardY
        dHRp_fy = dHRs_p(dRs[nx:],Rs[nx:],ko[nx:],visco[nx:],dbo[nx:],bo[nx:])*(1-backwardY)
        #derivative of Rs*oil in respect to saturation backward and forward in both directions.
        dHRs_bx, dHRs_fx = Rs[:-1]*dHso_bx, Rs[1:]*dHso_fx
        dHRs_bx[nx-1::nx], dHRs_fx[nx-1::nx] = np.zeros((ny-1,1)).flatten(), np.zeros((ny-1,1)).flatten()
        
        dHRs_by = Rs[:-nx]*dHso_by,
        dHRs_fy = Rs[nx:]*dHso_fy
                
        switcher ={
            "op": (dHpo_bx, dHpo_by, dHpo_fx, dHpo_fy),
            
            "sg": (dHsg_bx, dHsg_by, dHsg_fx, dHsg_fy),
            
            "so": (dHso_bx, dHso_by, dHso_fx, dHso_fy),
            
            "Rs_p": (dHRp_bx, dHRp_by, dHRp_fx, dHRp_fy),
            
            "gp": (dHpg_bx, dHpg_by, dHpg_fx, dHpg_fy),
            
            "Rs_s": (dHRs_bx, dHRs_by, dHRs_fx, dHRs_fy)
            }
        return switcher
        
    def upwinding_Jacob(self, P):
        """
        Constructing the upwinded derivatives. 'dHpo_fx' means the derivative of oil (transmissibility's 
        fluid part) in respect to pressure forward in the x direction, if 'dHpoy_fy' is the same definition 
        but forward in the y direction, 'dHRp_fx' is the derivative of Rs*H_oil (fluid part) in respect to
        pressure forward in the x direction, if 'b' id used in place of 'f' it means 'backward'. The same meaning 
        can be derived for the other expressions.
        
        Argument:
        P -- pressure vector.
        
        Return:
        The upwinded fluid part derivatives of oil and gas, in x and y direction and forward and backward.
        """
        #derivatives of upwinding oil transmissibilities, f is for forward, b is backward.
        #getting the upwinded derivatives.
        derivatives = self.derivate(P)
        
        dHpo_fx, dHpo_fy, dHpo_bx, dHpo_by = derivatives["op"]
        #With Rs
        dHRp_fx, dHRp_fy, dHRp_bx, dHRp_by = derivatives["Rs_p"] 
        #derivatives upwinding gas transmissibilities                                          
        dHpg_fx, dHpg_fy, dHpg_bx, dHpg_by = derivatives["gp"]
        #Doing the same thing for saturation.
        dHso_fx, dHso_fy, dHso_bx, dHso_by = derivatives["so"]
        #With Rs
        dHRs_fx, dHRs_fy, dHRs_bx, dHRs_by = derivatives["Rs_s"]
        #derivatives upwinding gas transmissibilities
        dHsg_fx, dHsg_fy, dHsg_bx, dHsg_by = derivatives["sg"]
                                                     
        return dHpo_fx, dHpo_fy, dHpo_bx, dHpo_by, dHRp_fx, dHRp_fy, dHRp_bx, dHRp_by,\
                    dHpg_fx, dHpg_fy, dHpg_bx, dHpg_by, dHso_fx, dHso_fy, dHso_bx, dHso_by,\
                            dHRs_fx, dHRs_fy, dHRs_bx, dHRs_by, dHsg_fx, dHsg_fy, dHsg_bx, dHsg_by
                        
        
    def derivatives_Jacob(self, d):
        """
        Constructing the derivative elements of the diagonals in Jacobian matrix from the upwinded fuid part
        derivatives. 'dUpO_fx' means the derivative of oil transmissibilities forward in the x direction,
        using 'fy' will mean forward in the y direction. 'dUsR_by' means th derivative of Rs*transmissibility
        in respect to saturation in the y direction backward.
        
        Argument:
        d -- a list containing all the deivatives of the upwinded fluid part (oil and gas, in x and y direction,
        backward and forward). This is the list we get from upwinding_jacob(P) function.
        """
        nx, ng = self.Nx, self.Nx*self.Ny
        #Constructing the transmissibilities derivatives (multiplying the fluid part with the goemtric part
        #using the diag() function:
        #for pressure derivatives
        dUpO_fx, dUpO_fy   = self.diag(d[0], "x", ng-1), self.diag(d[1], "y", ng-nx)
        dUpO_bx, dUpO_by   = self.diag(d[2], "x", ng-1), self.diag(d[3], "y", ng-nx)

        dUpR_fx, dUpR_fy = self.diag(d[4], "x", ng-1), self.diag(d[5], "y", ng-nx)
        dUpR_bx, dUpR_by = self.diag(d[6], "x", ng-1), self.diag(d[7], "y", ng-nx)
        
        dUpG_fx, dUpG_fy   = self.diag(d[8], "x", ng-1), self.diag(d[9], "y", ng-nx)
        dUpG_bx, dUpG_by   = self.diag(d[10], "x", ng-1), self.diag(d[11], "y", ng-nx)
        
        #for saturation derivatives                                            
        dUsO_fx, dUsO_fy   = self.diag(d[12], "x", ng-1), self.diag(d[13], "y", ng-nx)
        dUsO_bx, dUsO_by   = self.diag(d[14], "x", ng-1), self.diag(d[15], "y", ng-nx)

        dUsR_fx, dUsR_fy = self.diag(d[16], "x", ng-1), self.diag(d[17], "y", ng-nx)
        dUsR_bx, dUsR_by = self.diag(d[18], "x", ng-1), self.diag(d[19], "y", ng-nx)
        
        dUsG_fx, dUsG_fy   = self.diag(d[20], "x", ng-1), self.diag(d[21], "y", ng-nx)
        dUsG_bx, dUsG_by   = self.diag(d[22], "x", ng-1), self.diag(d[23], "y", ng-nx)
                                      
        return dUpO_fx, dUpO_fy, dUpO_bx, dUpO_by, dUpR_fx, dUpR_fy, dUpR_bx, dUpR_by, dUpG_fx, dUpG_fy,\
                    dUpG_bx, dUpG_by, dUsO_fx, dUsO_fy, dUsO_bx, dUsO_by, dUsR_fx, dUsR_fy,\
                        dUsR_bx, dUsR_by,dUsG_fx, dUsG_fy  ,dUsG_bx, dUsG_by  
        
    def diagonals_Jacob(self, der, P):
        """
        Constructing the diagonals which are going to be used in the Jacobian construction. 'diag_fx_po' means 
        the offset diagonals corresponding the oil derivative in respect to pressure forward in the x direction.
        'diag_by_sg' represent the diagonal for the gas derivative w.r.t to saturation backward in the y direction.
        
        Arguments:
        der -- a list containing the derivatives of transmissibilities from derivatives_jacob().
        P -- current pressure vector.
        
        Return:
        The offset diagonals which are going to be used in the Jacobian construction.
        """
        nx = self.Nx 
        diag1_o, diagNx_o, diag1_g, diagNx_g = self.diags
        #Constructing the jacobian's diagonals
        #Diagonals for pressure
        diag_fx_po = diag1_o
        diag_fx_po[1:] += der[0]*(P[:-2:] - P[2::])
        
        
        diag_fy_po = diagNx_o
        diag_fy_po[1:] += der[1]*(P[:-2*nx:] - P[2*nx::])
        
        diag_bx_po = diag1_o
        diag_bx_po[1:] -= der[2]*(P[:-2:] - P[2::])
        
        diag_by_po = diagNx_o
        diag_by_po[1:] -= der[3]*(P[:-2*nx:] - P[2*nx::])
        
        diag_fx_pg  = (der[4] + der[8])*(P[:-2:] - P[2::]) + diag1_g
        diag_fy_pg = (der[5] + der[9])*(P[:-2*nx:] - P[2*nx::]) + diagNx_g
        
        diag_bx_pg  = -(der[6] + der[10])*(P[:-2:] - P[2::]) + diag1_g
        diag_by_pg = -(der[7] + der[11])*(P[:-2*nx:] - P[2*nx::]) + diagNx_g
        
        #Diagonals for saturation
        diag_fx_so  = der[12]*(P[:-2:] - P[2::])
        diag_fy_so = der[13]*(P[:-2*nx:] - P[2*nx::])
        
        diag_bx_so  = -der[14]*(P[:-2:] - P[2::])
        diag_by_so = -der[15]*(P[:-2*nx:] - P[2*nx::])
        
        diag_fx_sg  = (der[16] + der[20])*(P[:-2:] - P[2::])
        diag_fy_sg = (der[17] + der[21])*(P[:-2*nx:] - P[2*nx::])
        
        diag_bx_sg  = -(der[18] + der[22])*(P[:-2:] - P[2::]) 
        diag_by_sg = -(der[19] + der[23])*(P[:-2*nx:] - P[2*nx::])
                                   
        return diag_fx_po,diag_fy_po,diag_bx_po,diag_by_po,diag_fx_pg,diag_fy_pg,diag_bx_pg,diag_by_pg,\
                    diag_fx_so,diag_fy_so,diag_bx_so,diag_by_so,diag_fx_sg,diag_fy_sg,diag_bx_sg,diag_by_sg                   
        
    def construct_Jacob(self, P, formulation):
        """
        Constructin the Jacobian using the helper functions to construct the diagonals first and then use
        these offset diagonals to construct the main ones.
        
        Argument: 
        P -- current pressure vector
        formulation -- FIM, IMPES,...
        Return:
        Updating the jacobian member variable and returning it.
        """
        nx, ng = self.Nx, self.Nx*self.Ny
        
        #upwinding the derivatives of the transmisibilitiy's fluid part 
        upwinded_diags = self.upwinding_Jacob(P) 
        #derivatives of the transmissibilities 
        derivatives = self.derivatives_Jacob(upwinded_diags)
        #constructing the diagonals                                       
        diag_fx_po,diag_fy_po,diag_bx_po,diag_by_po,diag_fx_pg,diag_fy_pg,diag_bx_pg,diag_by_pg,\
            diag_fx_so,diag_fy_so,diag_bx_so,diag_by_so,diag_fx_sg,diag_fy_sg,diag_bx_sg,diag_by_sg\
                = self.diagonals_Jacob(derivatives, P)                           
        
        #accumualtion terms
        diag0_accum, diag1_accum, diag_1_accum = self.accum_jacobian(P)
        #diagonals elements from wells:
        w_1, w0, w1 = self.diagWell[1][:-1], self.diagWell[2], self.diagWell[3][:-1]
        
        #main diagonals
        diag_o = -diag_1_accum[:-1] - w_1
        diag_o[1:] += diag_fx_so - diag_bx_po[:-1]
        diag_o[:-1] -= diag_fx_po[1:]
        diag_o[:-2*nx + 1] -= diag_fy_po[1:]
        diag_o[2*nx - 1:] -= diag_by_po[:-1]
            
        diag_g = - diag0_accum - w0
        diag_g[:-2]    -= diag_fx_pg
        diag_g[:-2*nx] -= diag_fy_pg
        diag_g[2:]     -= diag_bx_pg
        diag_g[2*nx:]  -= diag_by_pg
            
        diag_g[1:-1]      -= diag_fx_so
        diag_g[1:-2*nx+1] -= diag_fy_so
        diag_g[3:]        -= diag_bx_so[:-1]
        diag_g[2*nx+1:]   -= diag_by_so[:-1]
                                    
        diag_1 = diag_bx_po - diag1_accum[:-1] - w1
        diag_1[2:]  -= diag_bx_sg[:-1]
        diag_1[:-1] -= diag_fx_sg
        diag_1[:-2*nx + 1] -= diag_fy_sg
        diag_1[2*nx:]      -= diag_by_sg[:-1]
            
        #constrcuting the jacobian depending on the nonlinear formualtion
        if formulation == "IMPES":            
            update = (diag_g)[1::2]/(diag_1)[::2]
            self.residual[1::2] -= self.residual[::2]*update
            
            #The offset diagonals
            diag_f2, diag_f3 = diag_fx_pg.copy(), diag_fx_po[1:-1].copy()
            diag_f6, diag_f7 = diag_fy_pg.copy(), diag_fy_po[1:-1].copy()
                
            diag_b2, diag_b5 = diag_bx_pg.copy(), diag_by_po.copy()
            diag_b6 = diag_by_pg.copy()
            #Gauss elimination:
            diag_o[::2] -= update*diag_g[::2]
            diag_f3[::2] -= update[1:]*diag_f2[::2]
            diag_f7[::2] -= update[nx:]*diag_f6[::2]
              
            diag_g[1::2] -= diag_1[::2]*update
            diag_1[1::2] -= update[:-1]*diag_b2[::2]
            diag_b5[1::2] -= update[:-nx]*diag_b6[::2]
            
            
            self.jacobian = diags([diag_f2, diag_f3, diag_f6, diag_f7,\
                                       diag_b2, diag_b5, diag_b6,\
                                           diag_o, diag_g, diag_1],\
                                               [-2, -3, -2*nx, -2*nx-1, 2, 2*nx-1, 2*nx, -1, 0, 1],\
                                                   shape=(2*ng, 2*ng), format="csc") 
#            J.diagonal(-1)[1::2] = np.zeros((len(diag_o)//2))
            
#            d1 = (diag_1 - w1)[::2]
#            d0 = (diag_g - w0)[1::2]
#            
#            self.jacobian[::2,1::2] = diags(d1, 0)
#            self.jacobian[1::2,1::2] = diags(d0, 0)
#            J = self.jacobian
##            J.diagonal(-2)[1::2], J.diagonal(-2*nx+1)[1::2] = np.zeros((len(diag_f2)//2)), np.zeros((len(diag_f5)//2))
##            J.diagonal(-2*nx)[1::2] = np.zeros((len(diag_f6)//2))
##            
##            J.diagonal(2)[1::2], J.diagonal(3)[::2] = np.zeros((len(diag_b2)//2)), np.zeros((len(diag_b3)//2+1))
##            J.diagonal(2*nx)[1::2], J.diagonal(2*nx+1)[::2] = np.zeros((len(diag_b6)//2)), np.zeros((len(diag_b7)//2+1))
##            
##            J.diagonal(-1)[1::2] = np.zeros((len(diag_o)//2))
#
#            update = (diag_g)[1::2]/(diag_1)[::2]
#            self.residual[1::2] -= self.residual[::2]*update
#            self.jacobian[1::2,:] -= diags(update, 0)@self.jacobian[::2,:]
#            j = self.jacobian.todense()
            
            return self.jacobian[1::2,::2]
        
        elif formulation == "FIM":
            #The offset diagonals
            diag_f2, diag_f3, diag_f5 = diag_fx_pg + diag_fx_so, diag_fx_po[1:-1], np.pad(diag_fy_sg, (1,0), 'constant')
            diag_f6, diag_f7 = diag_fy_pg, diag_fy_po[1:-1]
            diag_f6[1:] += diag_fy_so[:-1]
                
            diag_b2, diag_b3, diag_b5 = diag_bx_pg, diag_bx_sg[:-1], diag_by_po
            diag_b6, diag_b7 = diag_by_pg + diag_by_so, diag_by_sg[:-1]
            diag_b2[1:] += diag_bx_so[:-1]
            
            self.jacobian = diags([diag_f2, diag_f3, diag_f5, diag_f6, diag_f7, \
                                       diag_b2, diag_b3, diag_b5, diag_b6, diag_b7, \
                                           diag_o, diag_g, diag_1], \
                                               [-2, -3, -2*nx+1, -2*nx, -2*nx-1, 2, 3, 2*nx-1, 2*nx, 2*nx+1, -1, 0, 1],\
                                                   shape=(2*ng, 2*ng), format="csc")            
            return self.jacobian
    
    def saturation_impes(self, pressure, Pn):
        b_o = self.vars["bo"]
        b_on = self.vars["bo_n"]
        return  1 - (((self.T@pressure)[1::2] - self.diagWell[0][1::2])/self.C.diagonal()[1::2] + b_on*(1-Pn[1::2]))/b_o
    
#            v, f = self.vars, self.functions
#        
#        W = Wells.wells
#        WI = W[:,2]
#        rates = W[:,0]
#        Pw = W[:,1]
#        BHPControl = W[:,3]
#        P = pressure[::2]
#        
#        visco = v['visco']
#        bo, ko = v['bo'], v['ko']
#        b_o, b_on = v["bo"], v["bo_n"]
#        
#        Tw = f['Tw']
#        
#        To = Tw(WI,ko,visco,bo)
#            
#        oil_Well = To*(P - Pw)*BHPControl + rates*(1 - BHPControl)
        
def fill_from_file(lines, params):
    for line in lines:
        g = open(line[1])
        params[line[0]] = []
        for l in g:
            params[line[0]].append(int(l))
        g.close()
        
        
#Initializing the parameters for the different classes.
def initialize_params(fileName):
    """
    The function responsible for initializing the different classes.
    
    Arguments:
    An input file describing all the parameters discribing the rock, fluid, grid and wells.
    
    Returns:
    parameters -- a dictionary containing all the created classes.
    """
    alpha = 0.001127                      #unit conversion constant
    #parameters
    params={}
    W={"pos_X": [], "pos_Y": [], "rates": [], "BHP": [], "Names": [], "diams": [], "Controls": []} #Well dictionary containing all the positions, rates and names
    try:
        f = open(fileName)
        for line in f:
            if line != '\n' and not line.startswith('#'):
                if line[0] != '$':
                    words = line.strip().split('=')
                    if words[0] == 'grid':
                        dX = f.readline().strip().split('=')
                        dY = f.readline().strip().split('=')
                        dZ = f.readline().strip().split('=')
                        if words[1] == 'heter':
                            fill_from_file((dX, dY, dZ), params)
                        else:
                            numCells = int(dX[1])*int(dY[1])*int(dZ[1])
                            for l in (dX, dY, dZ):
                                params[l[0]] = float(l[1])
                                params['d'+(l[0][1]).upper()]= np.array([int(params['L'+l[0][1]]/int(l[1])) for i in range(numCells)])
                    elif words[0][0] != 'K':
                        params[words[0]] = float(words[1])
                    elif words[0][-1] == 'h':
                        params[words[0][:-2]] = np.array([float(words[1]) for i in range(int(params["Nx"])*int(params["Ny"]))])
                    else:
                        fill_from_file([words], params)
                else:
                    wellName = line[1:].strip()
                    pos = f.readline().strip().split('=')[1].split()
                    rate = f.readline().strip().split('=')
                    BHP = f.readline().strip().split('=')
                    diam = f.readline().strip().split('=')
                    control = f.readline().strip().split('=')
                    W["pos_X"].append(int(pos[0]))
                    W["pos_Y"].append(int(pos[1]))
                    W["rates"].append(float(rate[1]))
                    W["Names"].append(wellName)
                    W["BHP"].append(float(BHP[1]))
                    W["diams"].append(float(diam[1]))
                    W["Controls"].append(control[1])
    finally:
        f.close()
        
    grid_dims = {"x": int(params['Nx']), "y": int(params['Ny']), "z": int(params['Nz'])} #grid dimensions on the three axis
    grid_discrt = {"x": params['dX'], "y": params['dY'], "z": params['dZ']}
    #initializing the different classes
    functions = {'visco': visc_o, 'viscg': visc_g, 'bo':b_o, 'bg':b_g, 'Rs':R_s, 'ko':k_ro, 'kg':k_rg,\
                 'dviscg':dvisc_g, 'dbo':db_o, 'dbg':db_g, 'dRs':dR_s, 'dko':dk_ro, 'dkg':dk_rg,\
                'dHop':dHop, 'dHgp':dHgp, 'dHs':dHs, 'dHRs_s':dHRs_s, 'dHRs_p':dHRs_p, 'H':H, 'HRs':HRs,\
                'T':T, 'Tw':Tw, 'dTop': dTop, 'dTgp': dTgp, 'dTs':dTs, 'dTgop':dTgop}
        
    rock = Rock(params['Cr'], params['Kx'], params['Ky'], params['Kz'], params['Phi'])
    oil = Fluids(functions['visco'], params['rho_o'], params['Co'])
    gas = Fluids(functions['viscg'], params['rho_g'], params['Cg'])
    grid = Grid(params['Lx'], params['Ly'], params['Lz'], rock, grid_dims, grid_discrt)
    wells = Wells(grid_dims, alpha)
    
    Pbub = np.full((grid_dims['x']*grid_dims['y']), params["Pbub"]) 
    Patm = np.full((grid_dims['x']*grid_dims['y']), params["Patm"])
    
    interfaces = Interfaces(grid_dims, rock, grid, alpha, functions, {"Pbub": Pbub, "Patm": Patm})
    
    #initializing the pressure vector with the initial pressure
    pressure = interweave(np.full((grid_dims['x']*grid_dims['y'], 1), params["Pinit"]), 
                    np.full((grid_dims['x']*grid_dims['y'], 1), params["Sg_init"]))       #indexing l = (j-1)*Nx + i
    
    wells.add_wells(W["pos_X"], W["pos_Y"], W["rates"], W["Names"], W["BHP"], W["diams"], W["Controls"],\
                            rock, grid)                                                   #adding the wells
    fluids = {'o':oil, 'g':gas}
    
    inputs = {"rock": rock, "grid": grid, "fluids": fluids, "wells": wells,
                  "grid_dims": grid_dims, "pressure": pressure, "interfaces": interfaces}
    return inputs

def wellViolated(wells, WBHP, T_control_change, t):
    """
    Checking if the one of the well conditions has been violated and if it is the case, change the condition from rate to BHP
    
    Arguments:
        wells -- the object containing the wells and their specs.
        P -- the pressure vector.
    Return:
        Bool indicating if a well condition has been violated.
    """
    flag = False
    wellPos = wells.wells_pos
    W = wells.wells
    for wellName,pos in wellPos.items():
        if(WBHP[wellName][-1] < W[pos,1] and not W[pos,3]):
            W[pos,3]=1
            T_control_change[wellName] = t
            flag = True
    return flag

def previous_step(avg_P, P_well_blocks, rates_well_blocks, WBHP, wells):
    avg_P.pop()
    for wellName in (wells.wells_pos).keys():
        P_well_blocks[wellName].pop()
        WBHP[wellName].pop()
        rates_well_blocks['o'][wellName].pop()
        rates_well_blocks['g'][wellName].pop()
    
def run_simulation(parameters, simulation_time, wellNames, etaP, etaS, w, dt, formulation):
    """
    The function responsible for running the simulation.
    
    Arguments:
    parameters -- the dictionary containing data from initialization.
    simulation_time -- for how long the simulation is goin to be run.
    wellNames -- an array of wellName that we are interested in following their pressure through time.
    etaP -- the desired pressure change in the time update.
    etaS -- the desired saturation change in the time update.
    dt -- the time step.
    
    Returns:
    plot of the pressure evolution with time in the well grid block.
    plot of the pressure at the last time step.
    """
    #unit conversion
    STB_to_MSCF = 5.615e-3
    #time variables
    t = 0
    #Maximum newton iterations before cutting the time step:
    kmax = 7
    #convergence criteria:
    eps1 = 1e-3
    eps2 = 1e-2
    eps3 = 1e-3
    #defining the different objects
    rock = parameters["rock"]
    wells = parameters["wells"]
    grid_dims = parameters["grid_dims"]
    
    fluids = parameters["fluids"]
    grid = parameters["grid"]
    interfaces = parameters["interfaces"]
    
    pressure = parameters["pressure"]
    wellsPos = wells.wells_pos
    W        = wells.wells 
    
    t_list = [0]
    pressure_well_blocks = {} #tracking the pressure at well blocks
    rates_well_blocks = {'o':{}, 'g':{}}
    avg_pressure = [pressure[0]]
    WBHP = {}
    
    time_steps = []
    iterations = [0]
    CFL = [0]
    #Will contain the time at which the Well control changes
    T_control_change = {}


    for wellName in wellNames: 
        pos = wellsPos[wellName]
        rates_well_blocks['o'][wellName] = [0]
        rates_well_blocks['g'][wellName] = [0]
        WBHP[wellName] = [pressure[0]]
        pressure_well_blocks[wellName] = [pressure[2*wellsPos[wellName]]]
        T_control_change[wellName] = 0
    i = 0
    dP = np.zeros_like(pressure)
    while t < simulation_time:
        Pn = pressure.copy()
        k  = 1
        while k < kmax:
            #updating variables and creating sparse matrices for transmissibility and accumulations:
            interfaces.update_variables(pressure, Pn)
            interfaces.update_compressibilities(grid, rock, fluids, dt)
            interfaces.update_transmissibilities(pressure)
            #Construcing the jacobian object and computing the residual and the Jacobian
            Jacob = Jacobian(interfaces, pressure, Pn, wells)
            #Flow rates for different wells
            source = Jacob.diagWell[0]
            To_1, Tgo, Tgg = Jacob.wellTrans
            J = Jacob.construct_Jacob(pressure, formulation)
            residual = Jacob.residual
            #Getting the variables we need to check for convergence
            bo, bg = interfaces.vars["bo"], interfaces.vars["bg"]
            B = 1/interweave(bg, bo)
            C = 1/interfaces.compressibilities.diagonal()
            #update the pressure vector
            if formulation == "FIM":
                dP = -spsolve(J, csc_matrix(residual).transpose())
                pressure += dP
                converged = (sc.linalg.norm(residual*C*B, np.inf) < eps1 and
                                     sc.linalg.norm(dP/np.mean(pressure[::2]), np.inf) < eps3 and
                                             sc.linalg.norm(dP[1::2], np.inf) < eps2)
            elif formulation == "IMPES":
                dPp = -spsolve(J, csc_matrix(residual[1::2]).transpose())
                pressure[::2] += dPp
                converged = (sc.linalg.norm((residual*C*B)[1::2], np.inf) < eps1 and
                                     sc.linalg.norm(dPp/np.mean(pressure[::2]), np.inf) < eps3)
                if converged:
                    interfaces.update_variables(pressure, Pn)
                    interfaces.update_compressibilities(grid, rock, fluids, dt)
                    interfaces.update_transmissibilities(pressure)
                    Jacob = Jacobian(interfaces, pressure, Pn, wells)
                    S = Jacob.saturation_impes(pressure, Pn)
                    dP[::2], dP[1::2] = dPp, S - Pn[1::2]
                    pressure[1::2] = S
            #Checking for convergence:
            if converged:
                avg_pressure.append(np.mean(pressure[::2]))
                #updating the well pressures and rates
                maxCFL = 0
                for wellName in wellNames:
                    pos = wellsPos[wellName]
                    pressure_well_blocks[wellName].append(pressure[2*wellsPos[wellName]])
                    if W[pos, 3] == 1:
                        #To vary BHP decomment the following line
                        #WBHP[wellName].append(W[pos,1]*(1 - 0.3*(t - T_control_change[wellName])/simulation_time))
                        WBHP[wellName].append(W[pos,1])
                        rates_well_blocks['o'][wellName].append(source[2*pos+1])
                        rates_well_blocks['g'][wellName].append(source[2*pos].flatten()*STB_to_MSCF)
                    else:
                        WBHP[wellName].append((pressure[2*wellsPos[wellName]] - W[pos,0]*To_1[pos]))
                        rates_well_blocks['o'][wellName].append(W[pos,0])
                        rates_well_blocks['g'][wellName].append((Tgo[pos] + Tgg[pos]).flatten()*To_1[pos]*W[pos,0]*STB_to_MSCF)
                    currCFL = (rates_well_blocks['o'][wellName][-1] + rates_well_blocks['g'][wellName][-1]/STB_to_MSCF)*C[0]
                    if currCFL > maxCFL:
                        maxCFL = currCFL
                CFL.append(maxCFL)
                break
            k+=1
            
        #cut the time step and restart from the previous time step. If well condt is violated, switch to BHP control
        if k == kmax or wellViolated(wells, WBHP, T_control_change, t):
            pressure = Pn.copy()
            dt /= 2
            if k!=kmax:
                previous_step(avg_pressure, pressure_well_blocks, rates_well_blocks, WBHP, wells)
        else:
            time_steps.append(dt)
            iterations.append(k)
            #updating time dt, deltaP is the pressure difference between time step n and n+1:
            if formulation == "FIM":
                deltaP = pressure - Pn
                dt *= min(np.min((1+w)*etaP/(np.array(np.abs(deltaP[::2])) + w*etaP)),\
                                      np.min((1+w)*etaS/(np.array(np.abs(deltaP[1::2])) + w*etaS)))
            #cut the time step if it exceeds 10
            if dt >= 15 : dt /= 2
            t += dt
            i+=1
            t_list.append(t)
            
    return {'BPR':pressure_well_blocks, 'FPR':avg_pressure, 'WBHP':WBHP, 'FGPR':rates_well_blocks['g'], 'FOPR':rates_well_blocks['o']},\
                    t_list, pressure, iterations, time_steps, CFL
def plot_comp(data, t, quantities, units, case):
    """
    Plotting the comparison figures between ECLIPSE and our simulator.
    
    Arguments:
        data -- a dict containing all the quantities from ECLIPSE and from our simulator.
        quantitities -- the desired quantities we want to compare.
        units -- the corresponding units
        case -- BHP control, rate control or mixture
    """
    for i, quant in enumerate(quantities):
        j=0
        for key, sim in data.items():
            if isinstance(sim[quant], dict):               
                for keyWell, values in sim[quant].items():
                    if key == 'Sim':
                        plt.plot(t, values, '--' + colors[j], label=quant + "--" + key)
                    else:
                        plt.plot(values, '--' + colors[j], label=quant + "--" + key)
            else:
                if key == 'Sim':   
                    plt.plot(t, sim[quant], '--' + colors[j], label=quant + "--" + key)
                else:
                    plt.plot(sim[quant], '--' + colors[j], label=quant + "--" + key)
            j+=1
        plt.title(quantities[i] + " vs time")
        plt.xlabel("Time (days)")
        plt.ylabel(quantities[i] + " (" + units[i] + ")")
        plt.grid()
        plt.legend()
        plt.savefig(quantities[i] + "_vs_time_" + case + "_test_test.png", dpi = 300)
        plt.show()
        
def plot_simulation(outputSim, t, quantities, units, case, plotname):
            for key, sim in outputSim.items():
                if isinstance(sim, dict):     
                    i = 0
                    for keyWell, values in sim.items():
                        plt.plot(t, values, '--' + colors[i%len(colors)], label=key + "--" + keyWell)
                        i+=1
                else:
                        plt.plot(t, sim, '--' + colors[0], label=key + "--" + key)
                plt.title(key + " vs time")
                plt.xlabel("Time (days)")
                plt.ylabel(key)
                plt.grid()
                plt.legend()
                plt.savefig(key + "_vs_time_" + case +"_" + plotname + "_.png", dpi = 300)
                plt.show()
        
def read_eclipse(excel_file, outputSim):
    """
    Reading the excel file and return a dict containing the quantities from our simulator and from Eclipse.
    
    Arguments:
        excel_file -- the file to read.
        outputSim -- the output from the run_simulation function containing all the desired quantities 
    """
    data = pd.read_excel(excel_file,  skiprows=4, index_col = 0)
    output = {}
    for quant in outputSim.keys():
        output[quant] = data[quant]
    return {'Sim': outputSim, 'Eclipse':output}

def plot_eclipse(excel_file, quantities, units):
    """
    Plotting excel data.
    
    Arguments:
        excel_file -- the file to read.
        quantities -- the quantities to plot.
        units -- the corresponding units to these quantities.
    """
    data = pd.read_excel(excel_file,  skiprows=4, index_col = 0)
    for i,quant in enumerate(quantities):
        plt.plot(data[quant], '--' + colors[i%len(colors)], label=quant)
        plt.title(quantities[i] + " vs time")
        plt.xlabel("Time (days)")
        plt.ylabel(quantities[i] + " (" + units[i] + ")")
        plt.grid()
        plt.legend()
        plt.savefig(quantities[i] + "_vs_time_.png", dpi = 800)
        plt.show()
        
def permeability_creation(filename, nx, ny, random, perm=None):
    f = open(filename, "w")
    for i in range(nx):
        for j in range(ny):
            if random:
                if j == -i + nx + 2:
                    perm = randint(1,2)
                elif j == -i + nx - 2:
                    perm = randint(10,11)
                else:
                    perm = randint(150,200)
            f.write(str(perm) + '\n')
    f.close()
    
        
#######################----------------------------------------------------------------------------#################################
##start of the main:##
####################################################################################################################################
start = time.time()  
etaP, etaS, w = 250, 0.25, 0.9
#Creating the permeability randomly if wanted
permeability_creation("permX.txt", 300, 300, False, 80)
permeability_creation("permY.txt", 300, 300, False, 120)
#initializing parameters:
parameters = initialize_params('parameters_53.txt')
#running the simulation: 
outputSimul, t, pressure, iterations, time_steps,_ = run_simulation(parameters, 1500, \
                                                ["Producer1"], etaP, etaS, w, 5, "IMPES")
Nx, Ny = parameters["grid_dims"]['x'], parameters["grid_dims"]['y']
#case variable: '1' if comparing to eclipse results for BHP control (change the control in the txt file parameters.txt)
#and '2' if case 2 (Rate + BHP) control. This shoudl match the control in the parameters file (if a comparison to ECLIPSE is wanted)
case = '2'
plotname = "Test_grid_83_" + case + ".png"

##### You can choose which plots to have, simulation alone, ECLIPSE alone, or a comparison of the two #####
#data = read_eclipse("Phase_4_"+case+".xlsx", outputSimul)
plot_simulation(outputSimul, t, ['BPR', 'FPR', 'WBHP', 'FOPR', 'FGPR'], ["Psi", "Psi", "Psi", "STB/day", "MSCF/day"], case, plotname)
#plot_comp(data, t, ['BPR', 'FPR', 'WBHP', 'FOPR', 'FGPR'], ["Psi", "Psi", "Psi", "STB/day", "MSCF/day"], case)
#plot_eclipse("Phase_4_1.xlsx", ['BPR', 'FPR', 'WBHP', 'FOPR', 'FGPR'], ["Psi", "Psi", "Psi", "STB/day", "MSCF/day"])

##plotting the pressure map at the last time step
#plt.figure(figsize=(7,6))
#pressure_map = sns.heatmap(pressure[::2].reshape((Nx, Ny)), cmap="RdBu_r")
#plt.xlabel("X axis")
#plt.ylabel("Y axis")
#plt.title("Pressure map")
#plt.show()
#figure = pressure_map.get_figure()    
#figure.savefig("Pressure_map_"+ plotname, dpi = 800)

print(time.time() - start)