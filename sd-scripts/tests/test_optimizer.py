from unittest.mock import patch
from library.train_util import get_optimizer
from train_network import setup_parser
import torch
from torch.nn import Parameter

# Optimizer libraries
import bitsandbytes as bnb
from lion_pytorch import lion_pytorch
import schedulefree

import dadaptation
import dadaptation.experimental as dadapt_experimental

import prodigyopt
import schedulefree as sf
import transformers


def test_default_get_optimizer():
    with patch("sys.argv", [""]):
        parser = setup_parser()
        args = parser.parse_args()
        params_t = torch.tensor([1.5, 1.5])

        param = Parameter(params_t)
        optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "torch.optim.adamw.AdamW"
        assert optimizer_args == ""
        assert isinstance(optimizer, torch.optim.AdamW)


def test_get_schedulefree_optimizer():
    with patch("sys.argv", ["", "--optimizer_type", "AdamWScheduleFree"]):
        parser = setup_parser()
        args = parser.parse_args()
        params_t = torch.tensor([1.5, 1.5])

        param = Parameter(params_t)
        optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
        assert optimizer_name == "schedulefree.adamw_schedulefree.AdamWScheduleFree"
        assert optimizer_args == ""
        assert isinstance(optimizer, schedulefree.adamw_schedulefree.AdamWScheduleFree)


def test_all_supported_optimizers():
    optimizers = [
        {
            "name": "bitsandbytes.optim.adamw.AdamW8bit",
            "alias": "AdamW8bit",
            "instance": bnb.optim.AdamW8bit,
        },
        {
            "name": "lion_pytorch.lion_pytorch.Lion",
            "alias": "Lion",
            "instance": lion_pytorch.Lion,
        },
        {
            "name": "torch.optim.adamw.AdamW",
            "alias": "AdamW",
            "instance": torch.optim.AdamW,
        },
        {
            "name": "bitsandbytes.optim.lion.Lion8bit",
            "alias": "Lion8bit",
            "instance": bnb.optim.Lion8bit,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW8bit",
            "alias": "PagedAdamW8bit",
            "instance": bnb.optim.PagedAdamW8bit,
        },
        {
            "name": "bitsandbytes.optim.lion.PagedLion8bit",
            "alias": "PagedLion8bit",
            "instance": bnb.optim.PagedLion8bit,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW",
            "alias": "PagedAdamW",
            "instance": bnb.optim.PagedAdamW,
        },
        {
            "name": "bitsandbytes.optim.adamw.PagedAdamW32bit",
            "alias": "PagedAdamW32bit",
            "instance": bnb.optim.PagedAdamW32bit,
        },
        {"name": "torch.optim.sgd.SGD", "alias": "SGD", "instance": torch.optim.SGD},
        {
            "name": "dadaptation.experimental.dadapt_adam_preprint.DAdaptAdamPreprint",
            "alias": "DAdaptAdamPreprint",
            "instance": dadapt_experimental.DAdaptAdamPreprint,
        },
        {
            "name": "dadaptation.dadapt_adagrad.DAdaptAdaGrad",
            "alias": "DAdaptAdaGrad",
            "instance": dadaptation.DAdaptAdaGrad,
        },
        {
            "name": "dadaptation.dadapt_adan.DAdaptAdan",
            "alias": "DAdaptAdan",
            "instance": dadaptation.DAdaptAdan,
        },
        {
            "name": "dadaptation.experimental.dadapt_adan_ip.DAdaptAdanIP",
            "alias": "DAdaptAdanIP",
            "instance": dadapt_experimental.DAdaptAdanIP,
        },
        {
            "name": "dadaptation.dadapt_lion.DAdaptLion",
            "alias": "DAdaptLion",
            "instance": dadaptation.DAdaptLion,
        },
        {
            "name": "dadaptation.dadapt_sgd.DAdaptSGD",
            "alias": "DAdaptSGD",
            "instance": dadaptation.DAdaptSGD,
        },
        {
            "name": "prodigyopt.prodigy.Prodigy",
            "alias": "Prodigy",
            "instance": prodigyopt.Prodigy,
        },
        {
            "name": "transformers.optimization.Adafactor",
            "alias": "Adafactor",
            "instance": transformers.optimization.Adafactor,
        },
        {
            "name": "schedulefree.adamw_schedulefree.AdamWScheduleFree",
            "alias": "AdamWScheduleFree",
            "instance": sf.AdamWScheduleFree,
        },
        {
            "name": "schedulefree.sgd_schedulefree.SGDScheduleFree",
            "alias": "SGDScheduleFree",
            "instance": sf.SGDScheduleFree,
        },
    ]

    for opt in optimizers:
        with patch("sys.argv", ["", "--optimizer_type", opt.get("alias")]):
            parser = setup_parser()
            args = parser.parse_args()
            params_t = torch.tensor([1.5, 1.5])

            param = Parameter(params_t)
            optimizer_name, _, optimizer = get_optimizer(args, [param])
            assert optimizer_name == opt.get("name")

            instance = opt.get("instance")
            assert instance is not None
            assert isinstance(optimizer, instance)
