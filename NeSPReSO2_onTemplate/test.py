import argparse

import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from model.loss import make_loss
from parse_config import ConfigParser
from base.util import prepare_device
from train import ensure_cache, set_seed
from types import SimpleNamespace


def main(config):
    logger = config.get_logger("test")
    set_seed(config.config.get("seed", 42))
    ensure_cache(config)

    dl_args = dict(config["data_loader"]["args"])
    dl_args["split"] = "test"
    dl_args["shuffle"] = False
    data_loader = getattr(module_data, config["data_loader"]["type"])(**dl_args)

    model = config.init_obj("arch", module_arch)
    logger.info(model)

    device, _ = prepare_device(config["n_gpu"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    pca_models = checkpoint.get("pca_models", data_loader.pca_models)
    outputs = checkpoint.get("outputs", dict(data_loader.outputs))

    density_meta = SimpleNamespace(
        LAT=data_loader.LAT,
        LON=data_loader.LON,
        PRES=data_loader.PRES,
        min_depth=data_loader.min_depth,
        max_depth=data_loader.max_depth,
    )
    loss_fn = make_loss(
        pca_models=pca_models,
        outputs=outputs,
        weights=data_loader.weights,
        device=device,
        density_config=config.config.get("density"),
        density_meta=density_meta,
        loss_config=config.config.get("loss_config"),
        targets=data_loader.cache["targets"],
        true_profiles=data_loader.cache.get("true_profiles"),
        ae_targets=data_loader.cache.get("ae_targets"),
        ae_weights=data_loader.cache.get("ae_weights"),
    )

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for data, target, indices in tqdm(data_loader):
            data = data.to(device)
            target = target.to(device)
            indices = indices.to(device)
            output = model(data)
            loss = loss_fn(output, target, indices)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, indices, data_loader) * batch_size

    n_samples = len(data_loader.dataset)
    log = {"loss": total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NeSPReSO v2 evaluation")
    args.add_argument("-c", "--config", default=None, type=str)
    args.add_argument("-r", "--resume", default=None, type=str, required=True)
    args.add_argument("-d", "--device", default=None, type=str)
    config = ConfigParser.from_args(args)
    main(config)
