import numpy as np
import torch

from repath.postprocess.slide_dataset import SlideDataset

def evaluate_on_multiple_devices(model, device, loader, num_classes):

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.eval()
    model.to(device)

    num_samples = len(loader) * loader.batch_size

    prob_out = np.zeros((num_samples, num_classes))

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, target = batch
            data = data.to(device)
            output = model(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes

            start = idx * loader.batch_size
            end = start + pred_prob.shape[0]
            prob_out[start:end, :] = pred_prob

            if idx % 100 == 0:
                print('Batch {} of {}'.format(idx, len(loader)))

    return prob_out


def evaluate_on_device(model, device, loader, num_classes, device_idx):

    model.eval()
    model.to(device)

    num_samples = len(loader) * loader.batch_size

    prob_out = np.zeros((num_samples, num_classes))

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, target = batch
            data = data.to(device)
            output = model(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes
            start = idx * loader.batch_size
            end = start + pred_prob.shape[0]
            prob_out[start:end, :] = pred_prob

            if idx % 100 == 0:
                print('Batch {} of {} on GPU {}'.format(idx, len(loader), device_idx))

    return prob_out
