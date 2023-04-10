"""
This training procedure is modification of
https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/simple_training.py
"""
import time

from nnverify.common import Domain
from nnverify import util
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler


def train(model, device, train_loader, criterion, optimizer, epoch, args, dataset):
    print(" ->->->->->->->->->-> One epoch with robust training <-<-<-<-<-<-<-<-<-<-")
    num_class = 10
    meter = MultiAverageMeter()

    ## wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    dummy_input, _ = next(iter(train_loader))
    lirpa_model = BoundedModule(model, dummy_input)

    eps_scheduler = FixedScheduler(args.epsilon)

    lirpa_model.train()
    eps_scheduler.train()
    eps_scheduler.step_epoch()
    eps_scheduler.set_epoch_length(int((len(train_loader) + args.batch_size - 1) / args.batch_size))

    torch.autograd.set_detect_anomaly(True)

    for i, (data, labels) in enumerate(train_loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        if eps < 1e-20:
            batch_method = "natural"
        if train:
            optimizer.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # bound input for Linf norm used only
        mean, std = util.get_mean_std(dataset)

        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        data_ub = torch.min(data + (eps / std).view(1, -1, 1, 1), data_max)
        data_lb = torch.max(data - (eps / std).view(1, -1, 1, 1), data_min)

        if list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        norm = np.inf
        if norm > 0:
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        elif norm == 0:
            ptb = PerturbationL0Norm(eps=eps_scheduler.get_max_eps(),
                                     ratio=eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
        x = BoundedTensor(data, ptb)

        output = lirpa_model(x)
        regular_ce = criterion(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('CE', regular_ce.item(), x.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0),
                     x.size(0))

        if args.trainer == Domain.LIRPA_IBP:
            lb, ub = lirpa_model.compute_bounds(IBP=True, C=c, method=None)
        elif args.trainer == Domain.LIRPA_CROWN:
            lb, ub = lirpa_model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
        elif args.trainer == Domain.LIRPA_CROWN_IBP:
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
            ilb, iub = lirpa_model.compute_bounds(IBP=True, C=c, method=None)
            if factor < 1e-5:
                lb = ilb
            else:
                clb, cub = lirpa_model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                lb = clb * factor + ilb * (1 - factor)
        elif args.trainer == Domain.LIRPA_CROWN_IBP:
            # Similar to CROWN-IBP but no mix between IBP and CROWN bounds.
            lb, ub = lirpa_model.compute_bounds(IBP=True, C=c, method=None)
            lb, ub = lirpa_model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
        else:
            raise ValueError('Unknown trainer')

        # Pad zero at the beginning for each example, and use fake label "0" for all examples
        lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
        fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
        robust_ce = criterion(-lb_padded, fake_labels)
        loss = robust_ce
        loss.backward()
        eps_scheduler.update_loss(loss.item() - regular_ce.item())
        optimizer.step()

        meter.update('Loss', loss.item(), data.size(0))
        meter.update('Robust_CE', robust_ce.item(), data.size(0))
        # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
        # If any margin is < 0 this example is counted as an error
        meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(epoch, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(epoch, i, eps, meter))