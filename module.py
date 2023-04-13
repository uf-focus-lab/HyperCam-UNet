import sys
import torch
import numpy as np
from tqdm import tqdm
from env import ensureDir, set_sigint, check_sigint
from score import score
from signal import signal, SIGINT


class Module(torch.nn.Module):

    activation = {}

    def __init__(self, device):
        super().__init__()
        self.device = device

    def run_train(self, train_loader, optimizer, lossFunction, epochs=1, kFoldRatio=1.0, report=True, file=sys.stdout, work_path=None):
        if work_path is not None:
            ensureDir(work_path)
        signal(SIGINT, set_sigint)
        for epoch in range(epochs):
            if check_sigint():
                return
            # Make the index start from 0
            epoch += 1
            # Progress bar
            prog = tqdm(train_loader, leave=False)
            prog.set_description("Epoch {:03d}".format(epoch))
            loss = None
            scores = []
            for batch, truth, _ in prog:
                if check_sigint():
                    return
                batch = batch.to(self.device)
                # Forward pass
                output = self(batch)
                # Release batch memory
                del batch
                # Compute truth
                truth = truth.to(self.device)
                # Record results
                scores.append(score(output, truth))
                # Compute loss
                loss = lossFunction(output, truth)
                del output
                del truth
                # Clear previously computed gradient
                optimizer.zero_grad()
                # Backward Propagation
                loss.backward()
                # Optimizing the parameters
                optimizer.step()
            loss = float(loss.detach())
            scores = np.average(scores, axis=1)
            # Report progress on demand
            if (report):
                msg = "Epoch {:03d} | Loss {:.6f} | AvgErr {:.6f} | StDev {:.6f}".format(epoch, loss, *scores)
                print(msg, file=file)
                if (file != sys.stdout):
                    print(msg, file=sys.stdout)

    def run_test(self, dataset, file=sys.stdout, work_path=None, report=True):
        if work_path is not None:
            ensureDir(work_path)
        with torch.no_grad():
            # weight_result = []
            prog = tqdm(dataset, leave=False)
            prog.set_description("Test Progress")
            for data, truth, name in prog:
                data = torch.stack([data.to(self.device)], dim=0)
                truth = torch.stack([truth.to(self.device)], dim=0)
                output = self(data)
                _, d, w, h = output.shape
                result = output.detach()
                result = result.view((d, w, h)).transpose(0, 2)
                result[result < 0] = 0
                result[result > 1] = 1
                result = torch.round(result * 255).type(torch.uint8)
                np.save(str(work_path / name), result.cpu().numpy())
                # Report progress on demand
                if (report):
                    msg = "Testing {} -> AvgErr {:.6f} | StDev {:.6f}".format(name, *score(output, truth))
                    print(msg, file=file)
