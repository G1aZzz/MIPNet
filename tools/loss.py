import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

class MseLoss:
    @staticmethod
    def calculate_loss(ests, gths, frames):
        EPSILON = 1e-7
        masks = []
        for frame in frames:
            masks.append(torch.ones(frame, ests.size()[2], dtype=torch.float32))
        masks = pad_sequence(masks, batch_first=True).cuda()
        ests = ests * masks
        gths = gths * masks
        loss = ((ests - gths) ** 2).sum() / masks.sum() + EPSILON
        return loss


class MaeLoss:
    @staticmethod
    def calculate_loss(ests, gths, frames):
        EPSILON = 1e-7
        masks = []
        for frame in frames:
            masks.append(torch.ones(frame, ests.size()[2], dtype=torch.float32))
        masks = pad_sequence(masks, batch_first=True).cuda()
        ests = ests * masks
        gths = gths * masks
        loss = (torch.abs(ests - gths)).sum() / masks.sum() + EPSILON
        return loss




class SiSnr:
    @staticmethod
    def calculate_loss(source, estimate_source):
        eps = 1e-8

        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)
        B, T = source.size()
        estimate = estimate_source[...,:T]
        source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
        dot = torch.matmul(estimate, source.t())  # B , B
        s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
        e_noise = estimate - source
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
        lo = 0 - torch.mean(snr)

        return lo

