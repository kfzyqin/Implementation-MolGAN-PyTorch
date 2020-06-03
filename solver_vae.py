from collections import defaultdict

import numpy as np
import os
import time
import datetime

import rdkit
import torch
import torch.nn.functional as F
from pysmiles import read_smiles
from torch.autograd import Variable
from torchvision.utils import save_image

from util_dir.utils_io import random_string
from utils import *
from models_vae import Generator, Discriminator, EncoderVAE
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
        self.f_dim = self.data.features
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_wgan = config.lambda_wgan
        self.lambda_rec = config.lambda_rec
        self.post_method = config.post_method

        self.metric = 'validity,qed'

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout_rate = config.dropout
        self.n_critic = config.n_critic
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

        # VAE KL weight.
        self.kl_la = 1.

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create an encoder and a decoder."""
        self.encoder = EncoderVAE(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.z_dim,
                                  with_features=True, f_dim=self.f_dim, dropout_rate=self.dropout_rate).to(self.device)
        self.decoder = Generator(self.g_conv_dim, self.z_dim, self.data.vertexes, self.data.bond_num_types,
                                 self.data.atom_num_types, self.dropout_rate).to(self.device)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.dropout_rate).to(self.device)

        self.vae_optimizer = torch.optim.RMSprop(list(self.encoder.parameters()) +
                                                 list(self.decoder.parameters()), self.g_lr)
        self.v_optimizer = torch.optim.RMSprop(self.V.parameters(), self.d_lr)

        self.print_network(self.encoder, 'Encoder', self.log)
        self.print_network(self.decoder, 'Decoder', self.log)
        self.print_network(self.V, 'Value', self.log)

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
        enc_path = os.path.join(self.model_dir_path, '{}-encoder.ckpt'.format(resume_iters))
        dec_path = os.path.join(self.model_dir_path, '{}-decoder.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(resume_iters))
        self.encoder.load_state_dict(torch.load(enc_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(dec_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.vae_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess_logits(inputs, method, temperature=1.):
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
        if self.resume_epoch:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        if self.mode == 'train':
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
                self.train_or_valid(epoch_i=i, train_val_test='sample')
        elif self.mode == 'test':
            assert self.resume_epoch is not None
            self.train_or_valid(epoch_i=start_epoch, train_val_test='sample')
            self.train_or_valid(epoch_i=start_epoch, train_val_test='val')
        else:
            raise NotImplementedError

    def get_reconstruction_loss(self, n_hat, n, e_hat, e):
        # This loss cares about the imbalance between nodes and edges.
        # However, in practice, they don't work well.
        # n_loss = torch.nn.CrossEntropyLoss(reduction='none')(n_hat.view(-1, self.m_dim), n.view(-1))
        # n_loss_ = n_loss.view(n.shape)
        # e_loss = torch.nn.CrossEntropyLoss(reduction='none')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
        # e_loss_ = e_loss.view(e.shape)
        # loss_ = e_loss_ + n_loss_.unsqueeze(-1)
        # reconstruction_loss = torch.mean(loss_)
        # return reconstruction_loss

        n_loss = torch.nn.CrossEntropyLoss(reduction='mean')(n_hat.view(-1, self.m_dim), n.view(-1))
        e_loss = torch.nn.CrossEntropyLoss(reduction='mean')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
        reconstruction_loss = n_loss + e_loss
        return reconstruction_loss

    @staticmethod
    def get_kl_loss(mu, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld_loss

    def get_gen_mols(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def get_reward(self, n_hat, e_hat, method):
        (edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols)).to(self.device)
        return reward

    def save_checkpoints(self, epoch_i):
        enc_path = os.path.join(self.model_dir_path, '{}-encoder.ckpt'.format(epoch_i + 1))
        dec_path = os.path.join(self.model_dir_path, '{}-decoder.ckpt'.format(epoch_i + 1))
        V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(epoch_i + 1))
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        torch.save(self.V.state_dict(), V_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    def get_scores(self, mols, to_print=False):
        scores = defaultdict(list)
        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        for k, v in m1.items():
            scores[k].append(v)
        for k, v in m0.items():
            scores[k].append(np.array(v)[np.nonzero(v)].mean())

        if to_print:
            log = ""
            is_first = True
            for tag, value in scores.items():
                if is_first:
                    log += "{}: {:.2f}".format(tag, np.mean(value))
                    is_first = False
                else:
                    log += ", {}: {:.2f}".format(tag, np.mean(value))
            print(log)
            return scores, log

        return scores

    def train_or_valid(self, epoch_i, train_val_test='val'):
        # Recordings
        losses = defaultdict(list)

        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = 1
            print('[Validating]')

        if train_val_test == 'sample':
            if self.mode == 'train':
                the_step = 1
            print('[Sampling]')

        for a_step in range(the_step):
            z = None
            if train_val_test == 'val':
                mols, _, _, a, x, _, f, _, _ = self.data.next_validation_batch()
            elif train_val_test == 'train':
                mols, _, _, a, x, _, f, _, _ = self.data.next_train_batch(self.batch_size)
            elif train_val_test == 'sample':
                z = self.sample_z(self.batch_size)
                z = torch.from_numpy(z).to(self.device).float()
            else:
                raise NotImplementedError

            if train_val_test == 'train' or train_val_test == 'val':
                a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
                x = torch.from_numpy(x).to(self.device).long()  # Nodes.
                a_tensor = self.label2onehot(a, self.b_dim)
                x_tensor = self.label2onehot(x, self.m_dim)
                f = torch.from_numpy(f).to(self.device).float()

            if train_val_test == 'train' or train_val_test == 'val':
                z, z_mu, z_logvar = self.encoder(a_tensor, f, x_tensor)
            edges_logits, nodes_logits = self.decoder(z)
            (edges_hat, nodes_hat) = self.postprocess_logits((edges_logits, nodes_logits), self.post_method)

            if train_val_test == 'train' or train_val_test == 'val':
                recon_loss = self.get_reconstruction_loss(nodes_logits, x, edges_logits, a)
                kl_loss = self.get_kl_loss(z_mu, z_logvar)
                loss_vae = recon_loss + self.kl_la * kl_loss

                # Real Reward
                reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
                # Fake Reward
                reward_f = self.get_reward(nodes_logits, edges_logits, 'hard_gumbel')

                # Value loss
                value_logit_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
                loss_v = torch.mean((value_logit_real - reward_r) ** 2 + (
                        value_logit_fake - reward_f) ** 2)
                loss_rl = torch.mean(-value_logit_fake)
                alpha = torch.abs(loss_vae.detach() / loss_rl.detach())
                loss_rl *= alpha

                vae_loss_train = self.lambda_wgan * loss_vae + (1 - self.lambda_wgan) * loss_rl
                # vae_loss_train = loss_vae
                losses['l_Rec'].append(recon_loss.item())
                losses['l_KL'].append(kl_loss.item())
                losses['l_VAE'].append(loss_vae.item())
                losses['l_RL'].append(loss_rl.item())
                losses['l_V'].append(loss_v.item())

                if train_val_test == 'train':
                    self.reset_grad()
                    vae_loss_train.backward(retain_graph=True)
                    loss_v.backward()
                    self.vae_optimizer.step()
                    self.v_optimizer.step()

            if train_val_test == 'sample':
                mols = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
                scores, mol_log = self.get_scores(mols, to_print=True)

                # Saving molecule images.
                mol_f_name = os.path.join(self.img_dir_path, 'sample-mol-{}.png'.format(epoch_i))
                save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

                if self.log is not None:
                    self.log.info(mol_log)

            if train_val_test == 'val':
                mols = self.get_gen_mols(nodes_logits, edges_logits, 'hard_gumbel')
                scores = self.get_scores(mols)

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

