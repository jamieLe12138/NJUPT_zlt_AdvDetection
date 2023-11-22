import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
import os
from os.path import join
def to_var(x, device='cuda',volatile=False):
    if volatile:
        x=x.to(device)
    else:
        with torch.no_grad():
            x=x.to(device)
    return x

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
    
class DAGMM(nn.Module):
    """Residual Block."""
    def __init__(self,in_channel,fSize,imSize,n_gmm = 2, latent_dim=3):
        super(DAGMM, self).__init__()

        self.in_channel=in_channel
        self.imSize=imSize
		
        inSize = imSize // (2 ** 4)
        self.encoder = nn.Sequential(nn.Conv2d(in_channel, fSize, 5, stride=2, padding=2),
                                     nn.Tanh(),
                                     nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2),
                                     nn.Tanh(),
                                     nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2),
                                     nn.Tanh(),
                                     nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2),
                                     nn.Tanh(),
                                     nn.Flatten(),
                                     nn.Linear((fSize * 8) * inSize * inSize, 1)
                                     )

        self.decoder = nn.Sequential(nn.Linear(1, (fSize * 8) * inSize * inSize),
                                     nn.Unflatten(dim=1,unflattened_size=((fSize * 8), inSize, inSize)),
                                     nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1),
                                     nn.Tanh(),
                                     nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1),
                                     nn.Tanh(),
                                     nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1),
                                     nn.Tanh(),
                                     nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)
                                    )


        self.estimation = nn.Sequential(nn.Linear(latent_dim,10),
                                        nn.Tanh(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(10,n_gmm),
                                        nn.Softmax(dim=1)
                                        )

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def relative_euclidean_distance(self, a, b):
        diff_norm = (a - b).norm(2, dim=1)
        # 计算 a 的 L2 范数
        a_norm = a.norm(2, dim=1)
        # 计算相对欧氏距离
        rel_distance = diff_norm / a_norm
        return rel_distance

    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)
        flatten_x=x.view(x.size(0),self.in_channel*self.imSize*self.imSize)
        flatten_dec=dec.view(dec.size(0),self.in_channel*self.imSize*self.imSize)
        rec_cosine = F.cosine_similarity(flatten_x, flatten_dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(flatten_x, flatten_dec)
        # print(rec_cosine.shape)
        # print(rec_euclidean.shape)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)
        # print(z.shape)
        # print(gamma.shape)


        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        # print(x.shape)
        # print(x_hat.shape)
        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
    
    def save_params(self, modelDir):
        print ('saving params...')
        torch.save(self.state_dict(), join(modelDir, 'DAGMM.pth'))


    def load_params(self, modelDir):
        print ('loading params...')
        self.load_state_dict(torch.load(join(modelDir, 'DAGMM.pth')))