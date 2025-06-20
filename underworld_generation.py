import numpy as np
import underworld as uw
import math
from underworld import function as fn
from underworld.scaling import units as u
from underworld.scaling import dimensionalise, non_dimensionalise
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import underworld.visualisation as vis
import os
import ipywidgets as widgets
from IPython.display import display

class ModelSetup:
    def __init__(self, width, height, x_coords=[0]):
        self.Lx        = non_dimensionalise( width * 1e3 * u.meter)
        self.Ly        = non_dimensionalise( height * 1e3 * u.meter)
        self.x_coords = x_coords
        
        self.gravity   = non_dimensionalise( 9.81 * u.meter / u.second**2)
        self.density   = non_dimensionalise( 3300 * u.kilogram / u.meter**3)
        self.viscosity = non_dimensionalise( 1e22 * u.Pa * u.sec)
        self.bulk_visc = non_dimensionalise( 1e11 * u.Pa *u.sec)
        
        self.mesh = uw.mesh.FeMesh_Cartesian( elementType = 'Q1/dQ0', 
                                         elementRes  = (width, height), 
                                         minCoord    = [0.,0.], 
                                         maxCoord    = [self.Lx,self.Ly] )
        
        self.bottomWall = self.mesh.specialSets["Bottom_VertexSet"]
        self.topWall    = self.mesh.specialSets["Top_VertexSet"]
        self.lateralWalls     = self.mesh.specialSets["Left_VertexSet"] + self.mesh.specialSets["Right_VertexSet"]
        
        self.velocityField = self.mesh.add_variable( nodeDofCount=self.mesh.dim )
        self.tractionField = self.mesh.add_variable( nodeDofCount=2 )
        self.pressureField = self.mesh.subMesh.add_variable( nodeDofCount=1 )
  
        self.initialize_scaling()
        self.initialize_swarms()
        

    def initialize_scaling(self):
        KL_meters = 100e3 * u.meter
        K_viscosity = 1e16 * u.pascal * u.second
        K_density = 3.3e3 * u.kilogram / (u.meter)**3
        KM_kilograms = K_density * KL_meters**3
        KT_seconds = KM_kilograms / (KL_meters * K_viscosity)
        K_substance = 1. * u.mole
        Kt_degrees = 1. * u.kelvin
        
        scaling_coefficients = uw.scaling.get_coefficients()
        scaling_coefficients["[length]"] = KL_meters.to_base_units()
        scaling_coefficients["[temperature]"] = Kt_degrees.to_base_units()
        scaling_coefficients["[time]"] = KT_seconds.to_base_units()
        scaling_coefficients["[mass]"] = KM_kilograms.to_base_units()
        
    def initialize_swarms(self):
        self.swarm = uw.swarm.Swarm( mesh=self.mesh, particleEscape=True )
        self.advector= uw.systems.SwarmAdvector(self.velocityField, self.swarm, order=2)
        
        self.materialVariable = self.swarm.add_variable( dataType="double", count=1 )
        
        swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=self.swarm, particlesPerCell=40 )
        self.swarm.populate_using_layout( layout=swarmLayout )
        
        self.materialVariable.data[:]=0
        for index,coord in enumerate(self.swarm.particleCoordinates.data):
            if coord[1] < self.Ly*0.6:
                self.materialVariable.data[index]=1
        
        self.population_control = uw.swarm.PopulationControl(self.swarm, 
                                                        aggressive=True,splitThreshold=0.15, maxDeletions=2,maxSplits=10,
                                                        particlesPerCell=40)

        self.mswarms = []  
        self.msAdvectors = []  
        for x in self.x_coords:
            mswarm = uw.swarm.Swarm(mesh=self.mesh, particleEscape=True)
            msAdvector = uw.systems.SwarmAdvector(self.velocityField, mswarm, order=2)
            
            particleCoordinates = np.zeros((1, 2))
            particleCoordinates[:, 0] = non_dimensionalise(x * 1e3 * u.meter)
            particleCoordinates[:, 1] = 0.6 * self.Ly
            ignore = mswarm.add_particles_with_coordinates(particleCoordinates)
            
            self.mswarms.append(mswarm)
            self.msAdvectors.append(msAdvector)

    def visualize(self, title="Uplift map", figsize=(700,400), quality=2, rulers=True, fn_size=2.):
        vdotv  = fn.math.dot(self.velocityField,self.velocityField)
        velmag = fn.math.sqrt(vdotv)
        
        cm_per_year = dimensionalise(1,u.centimeter/u.year)
        
        fig1 = vis.Figure(title=title, figsize=figsize, quality=quality, rulers=rulers)
        fig1.append( vis.objects.Points(self.swarm, self.materialVariable, fn_size=fn_size,colourBar = False  ) )
        fig1.append( vis.objects.VectorArrows(self.mesh, cm_per_year.magnitude*0.1*self.velocityField) )
        
        return fig1


class TractionSetup:
    def __init__(self, model, x_position, width, regionalfactor=0.6, nazcafactor=0.2):
        self.model = model
        self.x_position=x_position
        self.width=width
        self.regionalfactor=regionalfactor
        self.nazcafactor=nazcafactor
        
        self.xp = non_dimensionalise(x_position * 1e3 * u.meter)
        self.wi = non_dimensionalise(width * 1e3 * u.meter)
        lithostaticPressure = regionalfactor*self.model.Ly*self.model.density*self.model.gravity
        
        for ii in self.model.bottomWall:
            coord = self.model.mesh.data[ii]
            self.model.tractionField.data[ii] = [0.0, lithostaticPressure * (1. + nazcafactor * np.exp((-1 / self.wi * (coord[0] - self.xp)**2)))]

    def visualize(self):
        if uw.mpi.size == 1:
            plt.ion()
            pylab.rcParams['figure.figsize'] = 12, 6
            plt.title('Prescribed traction component normal to base wall')
            km_scaling = dimensionalise(1, u.kilometer)
            MPa_scaling = dimensionalise(1, u.MPa)
            plt.xlabel(f'X coordinate - (x{km_scaling.magnitude}km)')
            plt.ylabel(f'Normal basal traction MPa - (x{MPa_scaling.magnitude:.3e}MPa)')
            
            xcoord = self.model.mesh.data[self.model.bottomWall.data][:, 0]
            stress = self.model.tractionField.data[self.model.bottomWall.data][:, 1]
            
            plt.plot(xcoord, stress, 'o', color='black', label='numerical')
            plt.show()

class Simulation:
    def __init__(self, model, fig1, output_path = "upliftt/"):
        self.setup_directories(output_path)
        self.model=model
        
        lambdaFn = uw.function.branching.map( fn_key=self.model.materialVariable, 
                                            mapping={ 0: 1/self.model.bulk_visc, 1: 0.0 } )
        
        densityFn = uw.function.branching.map( fn_key=self.model.materialVariable, 
                                            mapping={ 0: 0.0, 1: self.model.density } )
        
        forceFn = densityFn * (0.0,-self.model.gravity)
        
        self.stokesBC = uw.conditions.DirichletCondition( variable      = self.model.velocityField, 
                                                     indexSetsPerDof = (self.model.lateralWalls, self.model.topWall) )
        
        nbc      = uw.conditions.NeumannCondition( fn_flux=model.tractionField, variable = self.model.velocityField, 
                                                     indexSetsPerDof = (None, self.model.bottomWall ) )
        
        self.stokesPIC = uw.systems.Stokes( velocityField = self.model.velocityField, 
                                       pressureField = self.model.pressureField,
                                       conditions    = [self.stokesBC, nbc],
                                       fn_viscosity  = self.model.viscosity, 
                                       fn_bodyforce  = forceFn,
                                       fn_one_on_lambda = lambdaFn )
        
        self.solver = uw.systems.Solver( self.stokesPIC )

    def simulate(self, steps=3):
        self.steps=steps
        
        dt    = -1.
        current_step = 0
        heights_at_x = []
        
        while current_step<steps:
            self.solver.solve()
            if dt < 0:
                dt = self.model.advector.get_max_dt()
                        
            self.model.advector.integrate(dt)              
            for ms_advector in self.model.msAdvectors:
                ms_advector.integrate(dt)
        
            heights_at_x.append(self.filter_heights_at_x())
                
            self.model.population_control.repopulate()
            fig1.save(self.output_path+"Uplift-"+str(current_step)+".png")
        
            current_step += 1        
        
        return heights_at_x
                    
    def filter_heights_at_x(self):
        heights = []
        for mswarm in self.model.mswarms: 
            fn_y = fn.input()[1] 
            fn_y_minmax = fn.view.min_max(fn_y)
            fn_y_minmax.evaluate(mswarm)
            heights.append(fn_y_minmax.max_global()) 
        return heights

    def animate_simulation(self, save=False):
        image_files = [self.output_path + f"Uplift-{i}.png" for i in range(self.steps)]
        images = []
        for fname in image_files:
            if os.path.exists(fname):
                images.append(plt.imread(fname))
            else:
                print(f"Warning: File not found {fname}")
                continue

        if not images:
            print("No images found to animate.")
            return

        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        img_plot = ax.imshow(images[0])

        def animate(i):
            img_plot.set_data(images[i])

        ani = FuncAnimation(fig, animate, frames=len(images), interval=200, repeat=True)
        plt.close()
        if save:
            gif_path = self.output_path + 'simulation_animation.gif'
            ani.save(gif_path, dpi=80, writer=PillowWriter(fps=5))
            print(f"Animation saved at {gif_path}")
        return ani

    def setup_directories(self, output_path):
        self.output_path = output_path
        try:
            if not os.path.exists("./"+output_path):
                os.makedirs("./"+output_path)
        except:
            raise
