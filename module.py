import sys
import torch
import numpy as np
from tqdm import tqdm
from env import ensureDir
from score import score


class Module(torch.nn.Module):

    activation = {}

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward_hook(self, layer, name):
        self.activation[name] = []

        def hook(model, input, output):
            self.activation[name].append(output.detach().cpu())
        layer.register_forward_hook(hook)

    def clear_activation(self):
        for name in self.activation:
            self.activation[name] = []

    def collect(self, activation_name):
        # return flatten(self.activation[activation_name])
        pass

    def run_train(self, train_loader, optimizer, lossFunction, epochs=1, kFoldRatio=1.0, report=True, file=sys.stdout, work_path=None):
        if work_path is not None:
            ensureDir(work_path)
        for epoch in range(epochs):
            # Make the index start from 0
            epoch += 1
            # Progress bar
            prog = tqdm(train_loader, leave=False)
            prog.set_description("Epoch {:03d}".format(epoch))
            loss = None
            scores = []
            for batch, truth, _ in prog:
                batch = batch.to(self.device)
                truth = truth.to(self.device)
                # Forward pass
                output = self(batch)
                # Record results
                scores.append(score(output, truth))
                # Compute loss
                loss = lossFunction(output, truth)
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
                result = output.view((d, w, h)).transpose(0, 2)
                result = torch.round(result * 255.0).type(torch.uint8)
                np.save(str(work_path / name), result.detach().cpu().numpy())
                # Report progress on demand
                if (report):
                    msg = "Testing {} -> AvgErr {:.6f} | StDev {:.6f}".format(name, *score(output, truth))
                    print(msg, file=file)
