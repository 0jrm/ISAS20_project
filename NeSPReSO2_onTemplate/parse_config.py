import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from base.util import read_json, write_json
from base.performance import validate_performance_config


def validate_config(config):
    """Trust-boundary checks before training or cache build."""
    outputs = config["outputs"]
    arch_type = config["arch"]["type"]
    arch_args = config["arch"]["args"]
    if arch_type == "FieldUNet":
        out_channels = arch_args.get("out_channels")
        if out_channels is not None:
            assert out_channels == sum(outputs.values()), (
                "out_channels must equal sum(outputs.values())"
            )
    else:
        output_dim = arch_args["output_dim"]
        assert output_dim == sum(outputs.values()), "output_dim must equal sum(outputs.values())"
    if arch_args.get("probabilistic"):
        nq = int(arch_args.get("n_quantiles") or 0)
        assert nq in (0, 9), "arch.args.n_quantiles must be 0 (hetero) or 9 (quantile)"
    assert int(config["io"]["spatial_pad"]) >= 0 and int(config["io"]["temporal_pad"]) >= 0
    density = config.get("density", {})
    if density.get("enabled"):
        assert os.path.isfile(density["checkpoint"]), f"missing density checkpoint: {density['checkpoint']}"
        assert os.path.isfile(density["stats_path"]), f"missing density stats: {density['stats_path']}"
    steric = config.get("steric") or {}
    if steric.get("enabled"):
        io = config.get("io", {}) or {}
        assert io.get("anomaly_targets") or io.get("steric_vs_sla"), (
            "steric.enabled requires io.anomaly_targets=true or io.steric_vs_sla=true"
        )
    loss_cfg = config.get("loss_config") or {}
    mode = loss_cfg.get("mode", "combined")
    from model.loss import VALID_LOSS_MODES

    assert mode in VALID_LOSS_MODES, f"loss_config.mode must be one of {VALID_LOSS_MODES}, got {mode!r}"
    if mode == "density_spice" and loss_cfg.get("prob_mode"):
        from model.loss import VALID_PROB_MODES

        assert loss_cfg["prob_mode"] in VALID_PROB_MODES, (
            f"loss_config.prob_mode must be one of {VALID_PROB_MODES}"
        )
        if loss_cfg["prob_mode"] in ("crps", "nll"):
            assert arch_args.get("probabilistic"), "crps/nll require arch.args.probabilistic=true"
            assert not arch_args.get("n_quantiles"), "crps/nll require n_quantiles=0"
        if loss_cfg["prob_mode"] == "quantile":
            assert int(arch_args.get("n_quantiles") or 0) == 9, "quantile mode requires n_quantiles=9"
        if loss_cfg.get("prob_mode") == "mse" and loss_cfg.get("freeze_sigma"):
            assert arch_args.get("probabilistic"), "freeze_sigma stage-1 needs probabilistic arch"
    if mode == "heave_residual":
        assert set(outputs) >= {"warp", "temperature", "salinity"}, (
            "heave_residual requires outputs warp+temperature+salinity"
        )
        if loss_cfg.get("prob_mode") in ("crps", "nll"):
            assert arch_args.get("probabilistic"), "heave_residual crps/nll require arch.args.probabilistic=true"
            assert not arch_args.get("n_quantiles"), "heave_residual crps/nll require n_quantiles=0"
    if mode == "profile_direct":
        assert set(outputs) == {"temperature", "salinity"}, (
            "profile_direct requires outputs {temperature, salinity}"
        )
        n_z = int(arch_args.get("n_z", 0))
        assert n_z > 1, "profile_direct requires arch.args.n_z"
        assert int(outputs["temperature"]) == n_z and int(outputs["salinity"]) == n_z, (
            "profile_direct outputs must equal n_z"
        )
        assert arch_type in ("LatentProfileDecoder", "ProfileDirect"), (
            "profile_direct requires LatentProfileDecoder or ProfileDirect"
        )
        assert not arch_args.get("probabilistic"), "profile_direct is deterministic (filter is the smoother)"
    if mode == "decoder":
        assert loss_cfg.get("decoder_dir"), "decoder mode requires loss_config.decoder_dir"
    if mode == "density_spice" or config.get("io", {}).get("representation") == "density_spice":
        assert set(outputs.keys()) == {"density_ctrl", "spice"}, (
            "density_spice requires outputs {density_ctrl, spice}"
        )
        assert int(outputs["spice"]) == 16, "density_spice expects spice=16 (PLAN §3.3)"
        n_scores = int(outputs["density_ctrl"])
        n_ctrl = int(config.get("io", {}).get("n_ctrl", 64))
        assert n_ctrl == 64, "density_spice expects io.n_ctrl=64 (PLAN §3.2 control grid)"
        assert 1 <= n_scores <= n_ctrl, (
            f"density_ctrl (R={n_scores}) must be in [1, n_ctrl={n_ctrl}]; "
            "R<K ⇒ low-rank δa (PLAN §3.6)"
        )
        assert mode == "density_spice", "representation=density_spice requires loss_config.mode=density_spice"
    if config.get("io", {}).get("representation") == "joint_eof":
        assert set(outputs.keys()) == {"joint"}, f"joint_eof requires outputs {{joint}}, got {list(outputs)}"
        assert int(outputs["joint"]) == 32, "joint_eof expects joint=32 (T1 / Phase 5 B)"
        assert mode in ("combined", "pc_mse_only", "pred_profile_cached"), (
            "joint_eof loss_config.mode must be combined|pc_mse_only|pred_profile_cached"
        )
    dl = config.get("data_loader", {}).get("args", {})
    split_mode = dl.get("split_mode", "random")
    assert split_mode in ("random", "chronological"), f"split_mode must be random|chronological, got {split_mode!r}"
    if split_mode == "chronological" and dl.get("split_config"):
        for split in ("train", "val", "test"):
            sc = dl["split_config"].get(split, {})
            assert sc.get("start") and sc.get("end"), f"split_config.{split} needs start/end dates"
    if config.get("performance"):
        from base.performance import get_performance_config

        validate_performance_config(get_performance_config(config))
    l3 = config.get("io", {}).get("l3")
    if l3 and l3.get("enabled"):
        for key in ("raw_root", "processed_root", "patch_half_deg", "grid_step_deg", "time_windows_hours"):
            assert key in l3, f"io.l3.enabled requires io.l3.{key}"
        assert len(l3["time_windows_hours"]) >= 1, "io.l3.time_windows_hours must be non-empty"
        from preproc.l3_rasterize import l3_geometry

        pad_s, pad_t, _ = l3_geometry(l3)
        assert int(config["io"]["spatial_pad"]) == pad_s, (
            f"io.spatial_pad ({config['io']['spatial_pad']}) must match l3 geometry ({pad_s})"
        )
        assert int(config["io"]["temporal_pad"]) == pad_t, (
            f"io.temporal_pad ({config['io']['temporal_pad']}) must match l3 geometry ({pad_t})"
        )
        from preproc.l3_rasterize import FEATURE_NAMES

        feats = l3.get("features", list(FEATURE_NAMES))
        assert feats, "io.l3.features must be non-empty"
        unknown = [f for f in feats if f not in FEATURE_NAMES]
        assert not unknown, f"io.l3.features unknown: {unknown}"
        vars_ = l3.get("variables") or {}
        assert vars_, "io.l3.variables must be non-empty"
    l4 = config.get("io", {}).get("l4")
    if l4 and l4.get("enabled"):
        from preproc.l4_augment import VALID_L4_MODES

        assert float(l4.get("noise_scale", 1.0)) >= 0, "io.l4.noise_scale must be >= 0"
    io = config.get("io", {})
    if io.get("dataset_tag") == "argo_cube" and io.get("refit_pca"):
        raise ValueError("argo_cube configs must set io.refit_pca=false (reuse point PCA basis)")
        mode = str(l4.get("mode", "mask_augment"))
        assert mode in VALID_L4_MODES, f"io.l4.mode must be one of {VALID_L4_MODES}, got {mode!r}"
        assert l3 and l3.get("enabled"), "io.l4.enabled requires io.l3.enabled"
    if "gsw_backend" in io:
        assert io["gsw_backend"] in ("gsw", "gsw_torch"), (
            f"io.gsw_backend must be 'gsw' or 'gsw_torch', got {io['gsw_backend']!r}"
        )
        from evalphys.gsw_backend import set_config_backend

        set_config_backend(io["gsw_backend"])


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        validate_config(config)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        run_id = getattr(args, "run_id", None)
        return cls(config, resume, modification, run_id=run_id)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '').replace('-', '_')
    return flags[0].replace('--', '').replace('-', '_')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
