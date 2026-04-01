import os
import time


from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)

def sample_1gaussian(n, scale=5, var=0.1):
    centers = torch.tensor([[0.0, 0.0]]) * scale

    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), math.sqrt(var) * torch.eye(2)
    )

    noise = m.sample((n,))
    return (centers[0] + noise).float()

def sample_2gaussians(n, scale=5, var=0.1):
    centers = torch.tensor([[1.0, 0.0], [-1.0, 0.0]]) * scale

    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(2), math.sqrt(var) * torch.eye(2)
    )

    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(2), n, replacement=True)

    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])

    return torch.stack(data).float()


sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_1gaussian(batch_size)
    x1 = sample_moons(batch_size)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_2gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj.cpu().numpy())
torch.save(model, f"{savedir}/otcfm_v2.pt")