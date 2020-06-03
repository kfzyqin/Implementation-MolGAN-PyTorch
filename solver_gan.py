from collections import defaultdict

import os
import time
import datetime

import rdkit
import torch
import torch.nn.functional as F
from pysmiles import read_smiles

from util_dir.utils_io import random_string
from utils import *
from models_gan import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""

        # Log
        self.log = log

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.la = config.lambda_wgan
        self.lambda_rec = config.lambda_rec
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        if self.la > 0:
            self.n_critic = config.n_critic
        else:
            self.n_critic = 1
        self.resume_epoch = config.resume_epoch

        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)

        # Directories.
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size.
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.dropout)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.dropout)

        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)
        self.v_optimizer = torch.optim.RMSprop(self.V.parameters(), self.g_lr)
        self.print_network(self.G, 'G', self.log)
        self.print_network(self.D, 'D', self.log)
        self.print_network(self.V, 'V', self.log)

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch is not None:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        if self.mode == 'train':
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
        elif self.mode == 'test':
            assert self.resume_epoch is not None
            self.train_or_valid(epoch_i=start_epoch, train_val_test='val')
        else:
            raise NotImplementedError

    def get_gen_mols(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def get_reward(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols)).to(self.device)
        return reward

    def save_checkpoints(self, epoch_i):
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(epoch_i + 1))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(epoch_i + 1))
        V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.V.state_dict(), V_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    def train_or_valid(self, epoch_i, train_val_test='val'):
        # The first several epochs using RL to purse stability (not used).
        if epoch_i < 0:
            cur_la = 0
        else:
            cur_la = self.la

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Iterations
        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = 1
            print('[Validating]')

        for a_step in range(the_step):
            if train_val_test == 'val':
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
            elif train_val_test == 'train':
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)
            else:
                raise NotImplementedError

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()  # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()

            # Current steps
            cur_step = self.num_steps * epoch_i + a_step
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute losses with real inputs.
            logits_real, features_real = self.D(a_tensor, None, x_tensor)

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # Compute losses for gradient penalty.
            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

            if cur_la > 0:
                losses['l_D/R'].append(d_loss_real.item())
                losses['l_D/F'].append(d_loss_fake.item())
                losses['l_D'].append(loss_D.item())

            # Optimise discriminator.
            if train_val_test == 'train' and cur_step % self.n_critic != 0 and cur_la > 0:
                self.reset_grad()
                loss_D.backward()
                self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # Value losses
            value_logit_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
            value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)

            # Feature mapping losses. Not used anywhere in the PyTorch version.
            # I include it here for the consistency with the TF code.
            f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

            # Real Reward
            reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
            # Fake Reward
            reward_f = self.get_reward(nodes_hat, edges_hat, self.post_method)

            # Losses Update
            loss_G = -logits_fake
            # Original TF loss_V. Here we use absolute values instead of the squared one.
            # loss_V = (value_logit_real - reward_r) ** 2 + (value_logit_fake - reward_f) ** 2
            loss_V = torch.abs(value_logit_real - reward_r) + torch.abs(value_logit_fake - reward_f)
            loss_RL = -value_logit_fake

            loss_G = torch.mean(loss_G)
            loss_V = torch.mean(loss_V)
            loss_RL = torch.mean(loss_RL)
            losses['l_G'].append(loss_G.item())
            losses['l_RL'].append(loss_RL.item())
            losses['l_V'].append(loss_V.item())

            alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
            train_step_G = cur_la * loss_G + (1 - cur_la) * alpha * loss_RL

            train_step_V = loss_V

            if train_val_test == 'train':
                self.reset_grad()

                # Optimise generator.
                if cur_step % self.n_critic == 0:
                    train_step_G.backward(retain_graph=True)
                    self.g_optimizer.step()

                # Optimise value network.
                if cur_step % self.n_critic == 0:
                    train_step_V.backward()
                    self.v_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Get scores.
            if train_val_test == 'val':
                mols = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
                m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
                for k, v in m1.items():
                    scores[k].append(v)
                for k, v in m0.items():
                    scores[k].append(np.array(v)[np.nonzero(v)].mean())

                # Save checkpoints.
                if self.mode == 'train':
                    if (epoch_i + 1) % self.model_save_step == 0:
                        self.save_checkpoints(epoch_i=epoch_i)

                # Saving molecule images.
                mol_f_name = os.path.join(self.img_dir_path, 'mol-{}.png'.format(epoch_i))
                save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

                # Print out training information.
                et = time.time() - self.start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                is_first = True
                for tag, value in losses.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                is_first = True
                for tag, value in scores.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                print(log)

                if self.log is not None:
                    self.log.info(log)

