import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

class MultiLossModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab.size, 300)
        self.drop = nn.Dropout(0.3)
        
        # Encoder for the content embedding
        self.contentE = nn.GRU(300, 256, 2,
            dropout=0.3 if 2 > 1 else 0, bidirectional=True)
        self.contentH2mu = nn.Linear(256*2, 128)
        self.contentH2logvar = nn.Linear(256*2, 128)

        # Encoder for the style embedding
        self.styleE = nn.GRU(300, 256, 2,
            dropout=0.3 if 2 > 1 else 0, bidirectional=True)
        self.styleH2mu = nn.Linear(256*2, 8)
        self.styleH2logvar = nn.Linear(256*2, 8)

        # Decoder
        self.z2emb = nn.Linear(128+8, 300)
        self.G = nn.GRU(300, 256, 2,
            dropout=0.3 if 2 > 1 else 0)
        self.proj = nn.Linear(256, vocab.size)

        # Sentence reconstruction
        self.opt = optim.Adam(self.parameters(), lr=0.001, betas=(0.5, 0.999))

        # Style classifier
        self.styleCls = nn.Linear(8, 2)

        # Content classifier
        self.contentCls = nn.Linear(128, 7604)

        # Content adverserial
        self.contentAdv = nn.Linear(8, 7604)

        # Style adverserial
        self.styleAdv = nn.Linear(128, 2)

    def contentEncode(self, input):
        input = self.drop(self.embed(input))
        _, h = self.contentE(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.contentH2mu(h), self.contentH2logvar(h)

    def styleEncode(self, input):
        input = self.drop(self.embed(input))
        _, h = self.styleE(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.styleH2mu(h), self.styleH2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def forward(self, input, is_train=False):
        # Encode content
        cont_mu, cont_logvar = self.contentEncode(input)
        content_z = reparameterize(cont_mu, cont_logvar)

        # Encode style
        style_mu, style_logvar = self.styleEncode(input)
        style_z = reparameterize(style_mu, style_logvar)

        z = content_z + style_z

        logits, _ = self.decode(z, input)

        style_preds = self.styleCls(style_mu)
        content_preds = self.contentCls(cont_mu)

        style_adv = self.styleAdv(cont_mu)
        content_adv = self.contentAdv(style_mu)

        return style_preds, content_preds, style_adv, content_adv, z, logits

    def loss_reconstruction(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def style_multitask_loss(self, style_pred, style_labels):
        loss = F.cross_entropy(style_pred, style_labels)
        return loss.sum(axis=0)

    def content_multitask_loss(self, content_pred, content_bow):
        loss = F.cross_entropy(content_pred, content_bow)
        return loss.sum(axis=0)
    
    def style_adverserial_loss(self, style_adv, style_labels):
        loss = F.cross_entropy(style_adv, style_labels)
        return loss.sum(axis=0)

    def content_adverserial_loss(self, content_adv, content_bow):
        loss = F.cross_entropy(content_adv, content_bow)
        return loss.sum(axis=0)
    
    def content_adv_batch_entropy(self, content_adv):
        return torch.mean(torch.sum(-content_adv * torch.log(content_adv + 1e-8)), 1)
    
    def style_adv_batch_entropy(self, style_adv):
        return torch.mean(torch.sum(-style_adv * torch.log(style_adv + 1e-8)), 1)
    
    def autoenc(self, inputs, targets, style_labels, content_bow):
        style_preds, content_preds, style_adv, content_adv, z, logits = self(inputs)
        return {'rec': self.loss_reconstruction(logits, targets).mean(),
                'style_mutitask': self.style_multitask_loss(style_preds, style_labels).mean(),
                'content_multitask': self.content_multitask_loss(content_preds, content_bow).mean(),
                'style_adverserial': self.style_adverserial_loss(style_adv, style_labels).mean(),
                'content_adverserial': self.content_adverserial_loss(content_adv, content_bow).mean(),
                'content_adverserial_entropy': self.content_adv_batch_entropy(content_adv),
                'style_adverserial_entropy': self.style_adv_batch_entropy(style_adv)}

    def step(self, losses):
        self.opt.zero_grad()
        for loss in losses:
            loss.backward()
        self.opt.step()
    
