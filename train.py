"""Training script.
usage: train.py [options]

options:
    --inner_learning_rate=ilr   Learning rate of inner loop [default: 2e-4]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-5]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 32]
    --length=l                  Length of image [default: 32]
    --dataset=ds                Dataset name (Mnist, Omniglot, VggFace, miniImageNet) [default: Mnist]
    --z_shape=zs                Dimension of latent code z [default: 128]
    --lambda_ms=lms             Lambda parameter of mode seeking regularization term [default: 1]
    --lambda_encoder=le         Lambda parameter of encoder loss term [default: 1]
    -h, --help                  Show this help message and exit
"""
from docopt import docopt


import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
import os
from datasets import MnistMetaEnv, OmniglotMetaEnv, VggFaceMetaEnv, miniImageNetMetaEnv
from model import Encoder, Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BCE_stable = nn.BCEWithLogitsLoss()
criterionL1 = nn.L1Loss()

# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		torch.nn.init.kaiming_normal_(m.weight)
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)
	elif classname.find('ConvTranspose2d') != -1:
		torch.nn.init.kaiming_normal_(m.weight)
		if m.bias is not None:
			m.bias.data.zero_()

class FAML:
    def __init__(self, args):
        self.load_args(args)
        self.id_string = self.get_id_string()
        self.writer = SummaryWriter('Runs/' + self.id_string)
        self.env = eval(self.dataset + 'MetaEnv(height=self.height, length=self.length)')
        self.initialize_gan()
        self.load_checkpoint()

    def inner_loop(self, real_batch):
        for p in self.meta_d.parameters():
            p.requires_grad = True
        
        self.meta_e.eval()
        x = real_batch

        for t in range(2):
            ########################
            # (1) Update D network #
            ########################

            self.meta_d.zero_grad()
            y_pred = self.meta_d(x)

            
            enc_x = self.meta_e(x)

            z1 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            z2 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)

            x_fake1, x_fake2 = self.meta_g(enc_x, z1), self.meta_g(enc_x, z2)

            pred_fake1 = self.meta_d(x_fake1.detach())
            pred_fake2 = self.meta_d(x_fake2.detach())

            errD1 = (BCE_stable(y_pred - torch.mean(pred_fake1), self.real_labels) + BCE_stable(pred_fake1 - torch.mean(y_pred), self.fake_labels))/2
            errD2 = (BCE_stable(y_pred - torch.mean(pred_fake2), self.real_labels) + BCE_stable(pred_fake2 - torch.mean(y_pred), self.fake_labels))/2

            errD = errD1 + errD2
            errD.backward()

            self.meta_d_optim.step()

        ########################
        # (2) Update G network #
        ########################
        # Make it a tiny bit faster
        for p in self.meta_d.parameters():
            p.requires_grad = False
        
        self.meta_e.train()
        
        for t in range(5):
            self.meta_g.zero_grad()
            self.meta_e.zero_grad()

            enc_x = self.meta_e(x)
            
            z1 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            z2 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)

            x_fake1, x_fake2 = self.meta_g(enc_x, z1), self.meta_g(enc_x, z2)
            
            pred_fake1 = self.meta_d(x_fake1)
            pred_fake2 = self.meta_d(x_fake2)

            y_pred = self.meta_d(x)
            
            errG1 = (BCE_stable(y_pred - torch.mean(pred_fake1), self.fake_labels) + BCE_stable(pred_fake1 - torch.mean(y_pred), self.real_labels))/2
            errG2 = (BCE_stable(y_pred - torch.mean(pred_fake2), self.fake_labels) + BCE_stable(pred_fake2 - torch.mean(y_pred), self.real_labels))/2

            errG = errG1 + errG2

            #### Train encoder
            l1_loss1 = criterionL1(x, torch.sigmoid(x_fake1))
            l1_loss2 = criterionL1(x, torch.sigmoid(x_fake2))

            errE = (l1_loss1 + l1_loss2) / 2

            # mode seeking loss
            lz = torch.mean(torch.abs(x_fake2 - x_fake1)) / torch.mean(torch.abs(z2 - z1))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            
            errG = errG + self.lms * loss_lz + 1 * self.le * errE

            errG.backward()

            self.meta_g_optim.step()
            self.meta_e_optim.step()

        self.decayD.step()
        self.decayG.step()
        self.decayE.step()

        return errD.item(), errG.item(), errE.item()

    def validation_run(self):
        data, task = self.env.sample_validation_task(self.batch_size)
        training_images = ((data-data.min()) / (data.max()-data.min())).cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        real_batch = data.to(device)

        d_total_loss = 0
        g_total_loss = 0
        e_total_loss = 0

        for _ in range(self.inner_epochs):
            d_loss, g_loss, e_loss = self.inner_loop(real_batch)
            d_total_loss += d_loss
            g_total_loss += g_loss
            e_total_loss += e_loss

        self.meta_g.eval()
        with torch.no_grad():
            enc_x = torch.cat((self.meta_e(real_batch), self.meta_e(real_batch), self.meta_e(real_batch)), dim=0)
            z = torch.randn((self.batch_size * 3, self.z_shape), dtype=torch.float, device=device)
            img = self.meta_g(enc_x , z)
        img = img.cpu().numpy()
        img = ((img-img.min()) / (img.max()-img.min()))
        img = np.concatenate([np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.batch_size)], axis=-1)
        img = np.concatenate([training_images, img], axis=-2)
        self.writer.add_image('Validation_generated', img, self.eps)
        self.writer.add_scalar('Validation_d_loss', d_total_loss, self.eps)
        self.writer.add_scalar('Validation_g_loss', g_total_loss, self.eps)
        self.writer.add_scalar('Validation_e_loss', e_total_loss, self.eps)

        print("Episode: {:.2f}\tD Loss: {:.4f}\tG Loss: {:.4f}\tG Loss: {:.4f}".format(self.eps, d_total_loss, g_total_loss, e_total_loss))

    def meta_training_loop(self):
        data, task = self.env.sample_training_task(self.batch_size)
        real_batch = data.to(device)

        d_total_loss = 0
        g_total_loss = 0
        e_total_loss = 0

        for _ in range(self.inner_epochs):
            d_loss, g_loss, e_loss = self.inner_loop(real_batch)
            d_total_loss += d_loss
            g_total_loss += g_loss
            e_total_loss += e_loss

        self.writer.add_scalar('Training_d_loss', d_total_loss, self.eps)
        self.writer.add_scalar('Training_g_loss', g_total_loss, self.eps)
        self.writer.add_scalar('Training_e_loss', e_total_loss, self.eps)

        # Updating both generator and dicriminator
        for p, meta_p in zip(self.g.parameters(), self.meta_g.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.g_optim.step()

        for p, meta_p in zip(self.d.parameters(), self.meta_d.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.d_optim.step()

        for p, meta_p in zip(self.e.parameters(), self.meta_e.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.e_optim.step()

    def reset_meta_model(self):
        self.meta_g.train()
        self.meta_g.load_state_dict(self.g.state_dict())

        self.meta_d.train()
        self.meta_d.load_state_dict(self.d.state_dict())

        self.meta_e.train()
        self.meta_e.load_state_dict(self.e.state_dict())

    def training(self):
        while self.eps <= 100000:
            self.reset_meta_model()
            self.meta_training_loop()

            # Validation run every 1000 training loop
            if self.eps % 1000 == 0:
                self.reset_meta_model()
                self.validation_run()
                self.checkpoint_model()
            self.eps += 1


    def load_args(self, args):
        self.outer_learning_rate = float(args['--outer_learning_rate'])
        self.inner_learning_rate = float(args['--inner_learning_rate'])
        self.batch_size = int(args['--batch_size'])
        self.inner_epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dataset = args['--dataset']
        self.z_shape = int(args['--z_shape'])
        self.lms = float(args['--lambda_ms'])
        self.le = float(args['--lambda_encoder'])

    def load_checkpoint(self):
        if os.path.isfile('Runs/' + self.id_string + '/checkpoint'):
            checkpoint = torch.load('Runs/' + self.id_string + '/checkpoint')
            self.d.load_state_dict(checkpoint['discriminator'])
            self.g.load_state_dict(checkpoint['generator'])
            self.e.load_state_dict(checkpoint['encoder'])
            self.eps = checkpoint['episode']
            print("Loading model from episode: ", self.eps)
        else:
            self.eps = 0

    def get_id_string(self):
        return '{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}'.format(self.dataset,
                                                                         str(self.outer_learning_rate),
                                                                         str(self.inner_learning_rate),
                                                                         str(self.batch_size),
                                                                         str(self.inner_epochs),
                                                                         str(self.height),
                                                                         str(self.length))

    def initialize_gan(self):
        # D and G on CPU since they never do a feed forward operation
        self.g = Generator(self.z_shape, self.height, self.env.channels)
        self.d = Discriminator(self.height, self.env.channels)
        self.e = Encoder(self.height, self.env.channels)
        self.meta_g = Generator(self.z_shape, self.height, self.env.channels).to(device)
        self.meta_d = Discriminator(self.height, self.env.channels).to(device)
        self.meta_e = Encoder(self.height, self.env.channels).to(device)
        self.g_optim = optim.Adam(params=self.g.parameters(), lr=self.outer_learning_rate, betas=(0.5, 0.999))
        self.d_optim = optim.Adam(params=self.d.parameters(), lr=self.outer_learning_rate, betas=(0.5, 0.999))
        self.e_optim = optim.Adam(params=self.e.parameters(), lr=self.outer_learning_rate, betas=(0.5, 0.999))
        self.meta_g_optim = optim.Adam(params=self.meta_g.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))
        self.meta_d_optim = optim.Adam(params=self.meta_d.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))
        self.meta_e_optim = optim.Adam(params=self.meta_e.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))

        self.real_labels = torch.ones(self.batch_size, dtype=torch.float, device=device)
        self.fake_labels = torch.zeros(self.batch_size, dtype=torch.float, device=device)

        decay = 0
        self.decayD = torch.optim.lr_scheduler.ExponentialLR(self.meta_d_optim, gamma=1-decay)
        self.decayG = torch.optim.lr_scheduler.ExponentialLR(self.meta_g_optim, gamma=1-decay)
        self.decayE = torch.optim.lr_scheduler.ExponentialLR(self.meta_e_optim, gamma=1-decay)

        self.meta_g.apply(weights_init)
        self.meta_d.apply(weights_init)
        self.meta_e.apply(weights_init)


    def checkpoint_model(self):
        checkpoint = {'discriminator': self.d.state_dict(),
                      'generator': self.g.state_dict(),
                      'encoder': self.e.state_dict(),
                      'episode': self.eps}
        torch.save(checkpoint, 'Runs/' + self.id_string + '/checkpoint')

if __name__ == '__main__':
    args = docopt(__doc__)
    env = FAML(args)
    env.training()