import os
import sys

ind_job = int(sys.argv[1]) - 1

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
import math
import os.path as osp
from tqdm import tqdm
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader

from dig.threedgraph.method.comenet.comenet import ComENet, EdgeGraphConv, Linear, TwoLayerLinear

# Modification of dig.threedgraph.dataset.QM93D
class GN3D(InMemoryDataset):

    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.url = ''
        self.mol_str = 'btbt'

        self.folder = osp.join(root, self.mol_str)
        self.data2, self.slices2 = (None, None)
        super(GN3D, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = (self.data2, self.slices2)
        self.data2, self.slices2 = (None, None)

    @property
    def raw_file_names(self):
        return self.mol_str+'_std.npz'

    @property
    def processed_file_names(self):
        return self.mol_str+'_pyg.pt'

    def download(self):
        # download_url(self.url, self.raw_dir)
        return

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names),allow_pickle=True)

        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        print(R.shape)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['homo','lumo','holu']:
            target[name] = np.expand_dims(data[name],axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['homo','lumo','holu']]
            data = Data(pos=R_i, z=z_i, y=y_i[0], homo=y_i[0], lumo=y_i[1], holu=y_i[2])

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        self.data2, self.slices2 = self.collate(data_list)

#         print('Saving...')
#         torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict



# Performs a forward pass of model, saves the per-atom energies to a csv file
def test(model,loader,fn_out):
    model.eval()
    score_out = []

    for data in loader:
        
        data = data.to(device)
        out, _, score = model(data)
        score_out.append(score[:,0].view(-1,32).detach().cpu().numpy())

    np.savetxt(fn_out,np.vstack(score_out),delimiter=',')
        
    return

# Performs a forward pass of model, saves the final prediction and reference value to a csv file
def testy(model,loader,fn_out):
    model.eval()
    score_out = []

    for data in loader:
        data = data.to(device)
        out, _, _ = model(data)
        out = torch.flatten(out)
        y = torch.flatten(data.y)
        score_temp = np.zeros((len(y),2))
        score_temp[:,0] = y.detach().cpu().numpy()
        score_temp[:,1] = out.detach().cpu().numpy()
        score_out.append(score_temp)

    np.savetxt(fn_out,np.vstack(score_out),delimiter=',')
        
    return

from torch_scatter import scatter
from torch_cluster import radius_graph
from torch.nn import Embedding

from torch_geometric.nn import GraphNorm
from torch import nn
import torch.nn.functional as F
from math import sqrt

r"""
This block of functions collects the subprocesses of the 
ComENet methods that we need to modify in order to implement CGMP.
CGMP-related modifications are found in the forward function for SelfInteractionBlock
as well as the two functions defined in ComENetPool.

"""
def swish(x):
    return x * torch.sigmoid(x)

class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels, _weight = 3/np.sqrt(hidden_channels)*torch.randn((95,hidden_channels)))

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.emb(x)
        return x

# The main modification needed is to include message gating the forward pass of the SIB
class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = torch.nn.LeakyReLU(.1)

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)


        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

#         self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

    def forward(self, x, gate_edge, feature1, feature2, edge_index, batch):
        
        feature1 = self.lin_feature1(feature1) 
        feature1 = feature1 * gate_edge 
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        feature2 = feature1 * gate_edge 
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))
        h = self.norm(h, batch)

        return h

class ComENetPool(ComENet):
    def __init__(
            self,
            cutoff=10,
            num_layers=3,
            hidden_channels=20,
            middle_channels=16,
            out_channels=1,
            num_radial=10,
            num_spherical=3,
            num_output_layers=3,
            pooling_ratio = .2
    ):
        super().__init__(cutoff=cutoff,num_layers=num_layers,hidden_channels=hidden_channels,
                         middle_channels=middle_channels,out_channels=out_channels,
                        num_radial=num_radial,num_spherical=num_spherical,num_output_layers=num_output_layers)
        act = swish
        self.num_atoms = 32
        self.emb = EmbeddingBlock(hidden_channels, act)
        self.emb2 = nn.Parameter(4*torch.ones(self.num_atoms,1)) # holds the node gate values
        self.node_bias = nn.Parameter(torch.zeros(1))
        self.edge_scale = 1
        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

    def edge_to_feat2(self,z,pos,edge_index,edge_score):
        j,i = edge_index
        
        vecs = pos[j] - pos[i]

        dist = vecs.norm(dim=-1)
        dist_inv = (dist**-2).view(-1,1) * edge_score # w_i
        r0_edge = dist_inv*vecs
        
        # Calculate differentiable reference vectors
        num_nodes = z.size(0)
        r0_sum = scatter(r0_edge, i, dim=0, dim_size=num_nodes) + 1e-6 * torch.randn(3,device=device)

        r0_rel = r0_sum[i] # f_i
        r1_edge = r0_edge - torch.sum(r0_rel * r0_edge,-1).view(-1,1) * r0_rel / (1e-9 + torch.sum(r0_rel*r0_rel,-1).view(-1,1))
        r1_sum = scatter(r1_edge, i, dim=0, dim_size=num_nodes) + 1e-2 * r0_sum + 1e-6 * torch.randn(3,device=device)
        r1_rel = r1_sum[i] # s_i
        
        
        r0j_rel = r0_sum[j]
        
        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            r0_rel,
            r1_rel,
            r0_rel,
            r0j_rel
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi
        
        return dist, theta, tau, phi
    
    def _forward(self, data):
        batch = data.batch
        z = data.z.long()
        pos = data.pos
        
        x1 = torch.tanh(self.emb(z))
        x2 = self.emb2
        x2_ref = torch.sigmoid(x2 + .5*torch.randn_like(x2) )
        x2 = x2_ref[z]
        x2_out = torch.sigmoid(self.emb2)

        
        pos2 = torch.cat((pos, 1.41*self.cutoff*(1-x2)*F.one_hot(torch.flatten(z),num_classes=self.num_atoms)),-1) 
        edge_index = radius_graph(pos2, r=self.cutoff, batch=batch)
        
        j,i = edge_index
        
        edge_node1 = x2[i]
        edge_node2 = x2[j]
        edge_score = (edge_node1 * edge_node2)
        m = nn.ReLU()
        dist, theta, tau, phi = self.edge_to_feat2(z,pos,edge_index,2*m(edge_score-.5))
        edge_score = edge_score**2
        
        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)
        
        x0 = x1 * x2 + torch.randn_like(x1) * 3 * (1 - x2) + torch.randn_like(x1) * .1
        x = x0
        for interaction_block in self.interaction_blocks:    
            x_int = interaction_block(x, edge_score, feature1, feature2, edge_index, batch) * x2
            x = x + x_int + torch.randn_like(x) * 3 * (1 - x2)
        
        for lin in self.lins:
            x = self.act(lin(x))
        x_int = self.lin_out(x)
        
        
        x_int = x_int + torch.randn_like(x_int) * .2 * (1 - x2)
        
        x = x_int * x2
        energy = scatter(x, batch, dim=0) + self.node_bias
        return energy, x2_out, x_int, x
    
    def forward(self, batch_data):
        out, out2, _, out3 = self._forward(batch_data)
        return out, out2, out3


from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Modification of dig.threedgraph.run
class Run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        self.l2 = 0 # loss tradeoff for atomic gates
        self.l3 = 0 # loss tradeoff for batch mean regularization
        self.target = 0 # target resolution during a training step
        self.n_atoms = 32
        pass
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, dir_out, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir=''):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """        

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        data_out = np.zeros((epochs,self.n_atoms))
        loss_out = np.zeros((epochs,6))
        
        target_list = np.ones(1000)*32
        for i in range(epochs):
            if i >= 100:
                target_list[i] = 32 - np.maximum(0,(i-100+30)//30)

        self.l2 = 10
        self.l3 = 1
        
        
        
        for epoch in range(1, epochs+1):
            self.target = target_list[epoch-1]
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            print('\nTraining...', flush=True)
            train_mae, train_loss_recon = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)

            print('\n\nEvaluating...', flush=True)
            valid_mae, valid_loss_recon = self.val(model, valid_loader, energy_and_force, p, evaluation, device)

            print('\n\nTesting...', flush=True)
            test_mae, test_loss_recon = self.val(model, test_loader, energy_and_force, p, evaluation, device)

            print()
            print({'Train': train_loss_recon, 'Validation': valid_loss_recon, 'Test': test_loss_recon})

            if log_dir != '':
                writer.add_scalar('train_mae', train_loss_recon, epoch)
                writer.add_scalar('valid_mae', valid_loss_recon, epoch)
                writer.add_scalar('test_mae', test_loss_recon, epoch)
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()
#             data_out[epoch-1,:] = torch.sqrt(torch.mean(model.emb2.emb.weight.data.detach()**2,1)).cpu().numpy()
            data_out[epoch-1,:] = model.emb2.data.detach()[:self.n_atoms,0].cpu().numpy()
            loss_out[epoch-1,0] = train_mae
            loss_out[epoch-1,1] = train_loss_recon
            loss_out[epoch-1,2] = valid_mae
            loss_out[epoch-1,3] = valid_loss_recon
            loss_out[epoch-1,4] = test_mae
            loss_out[epoch-1,5] = test_loss_recon
            
            if epoch == epochs:
                test(model,test_loader,dir_out+'/pool_test_'+str(int(evaluation))+'_'+str(int(self.target))+'.csv')
                testy(model,test_loader,dir_out+'/pool_testy_'+str(int(evaluation))+'_'+str(int(self.target))+'.csv')
            elif target_list[epoch] < target_list[epoch-1]:
                test(model,test_loader,dir_out+'/pool_test_'+str(int(evaluation))+'_'+str(int(self.target))+'.csv')
                testy(model,test_loader,dir_out+'/pool_testy_'+str(int(evaluation))+'_'+str(int(self.target))+'.csv')
        np.savetxt(dir_out+'/emb_progress_'+str(int(evaluation))+'.csv',data_out,delimiter=',')
        np.savetxt(dir_out+'/loss_progress_'+str(int(evaluation))+'.csv',loss_out,delimiter=',')
        print(f'Best validation loss so far: {best_valid}')
        print(f'Test loss when got best validation result: {best_test}')
        
        
        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.
        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        model.train()
        loss_accum = 0
        loss_recon_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            # prediction, gate values, atomic energies
            out, out2, out3 = model(batch_data)
            y_ref = batch_data.y.unsqueeze(1)
            loss_recon = loss_func(out, y_ref)
            loss_reg = ( self.l2 * (torch.sum(torch.abs(out2*(1.0-.01-out2))) / self.n_atoms 
                                     + 2 * self.n_atoms * ((torch.sum(out2)-self.target)/self.n_atoms)**2)
                        + self.l3 * torch.mean(torch.abs(torch.mean(torch.reshape(out3,(-1,self.n_atoms)),0)))
                        )
            loss = loss_recon + loss_reg
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
            loss_recon_accum += loss_recon.detach().cpu().item()
            

        return loss_accum / (step + 1), loss_recon_accum / (step + 1)

    def val(self, model, data_loader, energy_and_force, p, evaluation, device):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.
        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()

        loss_accum = 0
        loss_recon_accum = 0
        for step, batch_data in enumerate(tqdm(data_loader)):
            
            batch_data = batch_data.to(device)
            # prediction, gate values, atomic energies
            out, out3, out2 = model(batch_data)
            
            y_ref = batch_data.y.unsqueeze(1)
            loss_recon = loss_func(out, y_ref)
            loss_reg = ( self.l2 * (torch.sum(torch.abs(out2*(1.0-.01-out2))) / self.n_atoms 
                                     + 2 * self.n_atoms * ((torch.sum(out2)-self.target)/self.n_atoms)**2)
                        + self.l3 * torch.mean(torch.abs(torch.mean(torch.reshape(out3,(-1,self.n_atoms)),0)))
                        )
            loss = loss_recon + loss_reg
            loss_accum += loss.detach().cpu().item()
            loss_recon_accum += loss_recon.detach().cpu().item()

        return loss_accum / (step + 1), loss_recon_accum / (step + 1)
    



if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)
    print('device',device)

    dataset = GN3D(root='dataset/')
    target = 'homo'
    dataset.data.y = dataset.data[target]
    vt_batch_size = 32
    batch_size_train = 32
    
    # There are 14509 conformations in the full dataset file
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=12000, valid_size=509, seed=np.random.randint(0,100000))    
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train'][:((len(split_idx['train'])//batch_size_train)*batch_size_train)]], dataset[split_idx['valid'][:((len(split_idx['valid'])//batch_size_train)*batch_size_train)]], dataset[split_idx['test'][:((len(split_idx['test'])//batch_size_train)*batch_size_train)]]
    print('train, validation, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

    model = ComENetPool()
    model = model.to(device)

    loss_func = nn.MSELoss()
    dir_out = 'out_min/'
    run3d = Run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, dir_out, ind_job, epochs=1000, batch_size=32, vt_batch_size=64, lr=0.0025, lr_decay_factor=1, lr_decay_step_size=15, weight_decay=0)
    
