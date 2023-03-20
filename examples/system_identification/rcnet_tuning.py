# %%
"""|
System ID for networked dyanmical system. Toy problem for testing new class designs.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

#from neuromancer.psl.coupled_systems import *
from local_coupled_systems import *
import neuromancer.slim as slim

from neuromancer import blocks, estimators, integrators, ode, physics, dynamics
from neuromancer.interpolation import LinInterp_Offline
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.constraint import Variable, Loss
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.loss import PenaltyLoss
from neuromancer.activations import activations
from neuromancer.constraint import variable


from collections import OrderedDict
from abc import ABC, abstractmethod

# Local core development:
from local_interpolation import *

torch.manual_seed(0)
device = 'cpu'
#device = 'cuda:0'
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times"]
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})

params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
plt.rcParams.update(params)

# %% Set up the toy RC network:

# Adjacency list:
adj_dT = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T
adj_dT_symm = np.array([[0,1],[0,2],[0,3],[1,3],[1,4],[2,3],[3,4]]).T

# Physics definition:
nsim = 240
ts = 20.0
nx = 5
nu = 2
network = RC_Network_old(nsim=nsim, nx=nx, adj=adj_dT, ts=ts, periods=3)
#network = RC_Network(nsim=nsim, nx=nx, adj=adj_dT, ts=ts)
time = np.arange(nsim)*ts

# Simulation of ODE:
x0=(np.random.rand(5) * (10.0)) + 270.0
sim = network.simulate(x0=x0)

# %%
fig, ax = plt.subplots(1,2, figsize=(6.5,2))
ax=ax.flatten()
ax[0].plot(time,network.U[:,0],color=[0.3, 0.3, 0.3])
ax[0].set_title('Outdoor Temperature')
ax[0].set_ylabel('Kelvin')
ax[0].set_xlabel('Time [s]')
ax[1].set_title('Internal Heat Source')
ax[0].set_xlim([0,(nsim-1)*ts])
ax[1].plot(time,network.U[:,1],color=[0.3, 0.3, 0.3])
ax[1].set_ylim([0,20])
ax[1].set_xlim([0,(nsim-1)*ts])
ax[1].set_title('Internal Heat Source')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Power')

fig.tight_layout()
fig.savefig('forcing.png', format='png', dpi=300)
# %%
fig = plt.figure()
fig.text(0.35, 0.9, 'Room Temperatures Over Time')
locs = [(0,0),(0,2),(1,0),(1,1),(1,3)]
spans = [2,2,1,2,1]
ax = [plt.subplot2grid((2,4), l, colspan=s) for l,s in zip(locs, spans)]
[ax[i].plot(sim['X'][:,i]) for i in range(len(locs))];
miny = np.floor(np.min(sim['X']))
maxy = np.ceil(np.max(sim['X']))
[a.set_ylim(miny, maxy) for a in ax]
[a.tick_params(labelbottom=False) for a in ax]
[ax[i].tick_params(labelleft=False) for i in [1,3,4]]
fig.subplots_adjust(hspace=0.1,wspace=0.1)
plt.show()

# %% 
t = (np.arange(nsim+1) * ts).reshape(-1, 1)
U = network.U[:,0:nu]
sim['U'] = U
sim['Time'] = t
sim['U'] = np.concatenate((sim['U'][[0],:],sim['U']),axis=0)
plt.plot(sim['U'])
plt.plot(sim['X'][:,0])
plt.ylim([275,295])
# %%
nsteps = 1  # nsteps rollouts in training
nstep_data, loop_data, dims = get_sequence_dataloaders(sim, nsteps,
                            moving_horizon=True)

train_data, dev_data, test_data = nstep_data
train_loop, dev_loop, test_loop = loop_data

# Need to get t -> tensor, need interpolant
t = torch.from_numpy(t).to(device)
interp_u = Piecewise(t,(torch.from_numpy(U).to(device)))

# %% Helper function:
def generate_parameterized_edges(physics,edge_list):
    """
    Quick helper function to construct edge physics/objects from adj. list:
    """

    couplings = []
    if isinstance(physics,nn.Module): # is "physics" an instance or a class?
        # If we're in here, we expect one instance of "physics" for all edges in edge_list (homogeneous edges)
        physics.pins = edge_list
        couplings.append(physics)
        print(f'Broadcasting {physics} to all elements in edge list.')
    else: 
        # If we're in here, we expect different "physics" for each edge in edge_list (heterogeneous edges)
        for edge in edge_list:
            agent = physics(R=nn.Parameter(torch.tensor([1.0])),pins=[edge],symmetric=True)
            couplings.append(agent)

        print(f'Assuming new {physics} for each element in edge list.')

    return couplings

# %% Instantiation/Model construction
adjacency = list(adj_dT_symm.T)                                              # Interaction physics
outside = physics.SourceSink()                                                  # Constant/non-constant outdoor temp?
heater = physics.SourceSink()                                                   # Constant/non-constant outdoor temp?

# Define agents in network:
agents = [physics.RCNode(C=nn.Parameter(torch.tensor([1.0])),scaling=1.0e-3) for i in range(5)]  # heterogeneous population w/ identical physics
#room = RCNode(C=nn.Parameter(torch.tensor(5.0), requires_grad = False),scaling=1.0e-5)
#agents = [room for i in range(5)]  # heterogeneous population w/ identical physics
agents.append(outside) # Agent #5
agents.append(heater) # Agent #6
map = physics.map_from_agents(agents)                                           # Construct state mappings


#agents[0].C = C=nn.Parameter(torch.tensor(1.0), requires_grad=False)

# Define inter-node couplings:
couplings = generate_parameterized_edges(physics.DeltaTemp,adjacency)           # Heterogeneous edges of same physics

# Couple w/ outside temp:
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor([1.0])),pins=[[0,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor([1.0])),pins=[[1,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor([1.0])),pins=[[2,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor([1.0])),pins=[[3,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor([1.0])),pins=[[4,5]]))

# Couple w/ hvac:
couplings.append(physics.HVACConnection(pins=[[0,6],[1,6],[2,6],[3,6],[4,6]]))

# Define ODE system:
model_ode = ode.GeneralNetworkedODE(
    map = map,
    agents = agents,
    couplings = couplings,
    insize = nx+nu,
    outsize = nx,
    inductive_bias="compositional")


# %%
fy = slim.maps['identity'](nx, nx)

fx_int = integrators.RK4(model_ode, interp_u = interp_u, h=ts)
#fx_int = integrators.DiffEqIntegrator(model_ode, interp_u = interp_u, h=ts)

estim = estimators.FullyObservable(
    {**train_data.dataset.dims, "x0": (nx,)},
    linear_map=slim.maps['identity'],
    input_keys=["Yp"],
)

dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                input_key_map={"x0": f"x0_{estim.name}", "Time": "Timef", 'Yf': 'Yf'},  #TBC2: sth wrong with input_key_map
                name='dynamics',  # must be named 'dynamics' due to some issue in visuals.py
                online_flag=False
                )

#dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
#                input_key_map={"x0": f"x0_{estim.name}", 'Yf': 'Yf'},  #TBC2: sth wrong with input_key_map
#                name='dynamics',  # must be named 'dynamics' due to some issue in visuals.py
#                online_flag=False
#                )

# %%
yhat = variable(f"Y_pred_{dynamics_model.name}")
y = variable("Yf")

reference_loss = ((yhat == y)^2)
reference_loss.name = "ref_loss"

yFD = (y[:, 1:, :] - y[:, :-1, :])
yhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])

FD_loss = ((yhat == y)^2)
FD_loss.name = "ref_loss"

objectives = [reference_loss]
constraints = []
components = [estim, dynamics_model]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)

# %%
optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(problem.parameters(), lr=1.0)
#optimizer = torch.optim.RMSprop(problem.parameters(), lr=0.1)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout="nstep_dev_"+reference_loss.output_keys[0])

simulator = OpenLoopSimulator(
    problem, train_loop, dev_loop, test_loop, eval_sim=True, device=device,
) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
    problem, train_loop, dev_loop, test_loop, eval_sim=True, device=device,
)
visualizer = VisualizerOpen(
    dynamics_model,
    1,
    'test',
    training_visuals=False,
    trace_movie=False,
)
callback = SysIDCallback(simulator, visualizer)

trainer = Trainer(
    problem,
    train_data,
    dev_data,
    test_data,
    optimizer,
    callback=callback,
    epochs=5000,
    patience=20,
    warmup=5,
    #eval_metric="nstep_dev_"+reference_loss.output_keys[0],
    eval_metric="nstep_dev_loss",
    train_metric="nstep_train_loss",
    dev_metric="nstep_dev_loss",
    test_metric="nstep_test_loss",
    logger=logger,
    device=device,
)
# %%
best_model = trainer.train()

# %%
best_outputs = trainer.test(best_model)

# %%
plt.style.use('default')

plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times"]
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})

params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
plt.rcParams.update(params)


# %%
fx_int = integrators.RK4(model_ode, interp_u = interp_u, h=ts)
end_step = nsim+50
sol = torch.zeros((end_step,5))
ic = torch.unsqueeze(torch.tensor(x0),0).float()
t = 0
for j in range(sol.shape[0]-1):
    if j==0:
        sol[[0],:] = ic.float()
        sol[[j+1],:] = fx_int(ic,None,t)
    else:
        sol[[j+1],:] = fx_int(sol[[j],:],None,t)
    t += ts

# %% Visualize results:

fig, axs = plt.subplots(5,1, figsize=(5,17))

for idx in range(5):
    axs[idx].plot(sol[:,idx].detach().numpy(),label='model')
    axs[idx].plot(sim['X'][:,idx],label='actual')
    axs[idx].legend()
    axs[idx].set_title(f'Node #{idx+1}')
    axs[idx].set_ylim([275,292])
    axs[idx].set_xlim([0,end_step])
    axs[idx].plot([nsim,nsim],[0,1e3])    
    
# %%
plt.plot(sol.detach().numpy())
plt.title('Model')
plt.xlim([0,end_step])

# %%
plt.plot(sim['X'])
plt.title('Actual')
plt.xlim([0,end_step])

# %%
fig, axs = plt.subplots(3,1, figsize=(5,9))

for idx in range(3):
    axs[idx].plot(sol[:,idx].detach().numpy(),label='model')
    axs[idx].plot(sim['X'][:,idx],label='actual')
    axs[idx].legend()
    axs[idx].set_title(f'Node #{idx+1}')
    axs[idx].set_ylim([275,292])
    axs[idx].set_xlim([0,end_step])
    axs[idx].plot([nsim,nsim],[0,1e3])    
# %%
time = np.arange(nsim)*ts

fig, axs = plt.subplots(3,2, figsize=(6.5,7))
axs=axs.flatten()
for idx in range(5):
    axs[idx].plot(time,sim['X'][:nsim,idx],label='Actual',color=[0.3,0.3,0.8],ls='--')
    axs[idx].plot(time,sol[:nsim,idx].detach().numpy(),label='Model', color=[0.8,0.3,0.3])
    axs[idx].set_title(f'Node #{idx+1}')
    axs[idx].set_ylim([275,287.5])
    axs[idx].set_xlim([0,nsim*ts - ts])
    axs[idx].set_xlabel('Time [s]')
    axs[idx].set_ylabel('Temperature [K]')
    axs[idx].plot([(nsim/3)*ts,(nsim/3)*ts],[0,1e3],color=[0.3,0.3,0.3])
    if idx == 0:
        axs[idx].legend()

axs[-1].axis('off')
fig.tight_layout()

fig.savefig('results.png', format='png', dpi=300)
# %% Section for determining time constants for agents
# Need to look at both from psl and from trained model:

# define the time constant to be:
# tau_i =  C_i*(sum_Aij(Rj))
adj = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]])
#network.R
#network.C
N = [3,3,2,4,2,1]
tau = {}

# Agent-coupling contributions:
for i,pair in enumerate(list(adj)):
    if not (str(pair[0]) in tau.keys()):
        tau[str(pair[0])] = 0

    tau[str(pair[0])] += network.R[i]

# External coupling contributions and scale by capacitance:
#for key in tau.keys():
#    tau[key] = network.C[int(key)]*(tau[key] + network.R_ext[int(key)])

for key in tau.keys():
    tau[key] = network.C[int(key)]*((tau[key] + network.R_ext[int(key)])/N[int(key)])
# %% Now for the model's time constants:

tau_model = {}

for i,coupling in enumerate(model_ode.couplings):
    # only look at dT connections
    if not isinstance(coupling,physics.DeltaTemp):  
        print(coupling)
        continue
    
    for pin in coupling.pins:
        # are the agents in the dict? if not, add them and set to zero
        if not (str(pin[0]) in tau_model.keys()):
            tau_model[str(pin[0])] = 0

        if not (str(pin[1]) in tau_model.keys()):
            tau_model[str(pin[1])] = 0

        tau_model[str(pin[0])] += 1.0/coupling.R.item()
        tau_model[str(pin[1])] += 1.0/coupling.R.item()

for key in tau_model.keys():
    tau_model[key] = tau_model[key]/N[int(key)]

for i,agent in enumerate(model_ode.agents):
    if not isinstance(agent,physics.RCNode):  
        continue
    print(1.0/(agent.scaling*agent.C.item()))
    tau_model[str(i)] *= 1.0/(agent.scaling*agent.C.item())


tau = np.array([*tau.values()])
tau_model = np.array([*tau_model.values()])

errors = (tau-tau_model[:5])/tau
# %%

fig, ax = plt.subplots(1,1,figsize=(3.5,2))

ax.scatter([0,1,2,3,4],tau,label="Actual", color=[0.3,0.3,0.3], marker=',')
ax.scatter([0,1,2,3,4],tau_model[:5],label="Model", color=[0.7,0.7,0.7], marker='.')
ax.set_xticks([0, 1, 2, 3, 4], ['N1', 'N2', 'N3', 'N4', 'N5'],
       rotation=20)
ax.set_ylim(0,2500)
ax.set_xlabel('Node number')
ax.set_ylabel('Time constant [s]')
ax.legend()
fig.tight_layout()
fig.savefig('time_constants.png', format='png', dpi=300)

# %%
