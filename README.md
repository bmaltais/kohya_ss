# Kohya's dreambooth and finetuning

This repository now includes the solutions provided by Kohya_ss in a single location. I have combined both solutions under one repository to align with the new official Kohya repository where he will maintain his code from now on: https://github.com/kohya-ss/sd-scripts.

A note accompanying the release of his new repository can be found here: https://note.com/kohya_ss/n/nba4eceaa4594

## Dreambooth

You can find the dreambooth solution spercific [Dreambooth README](README_dreambooth.md)

## Finetune

You can find the finetune solution spercific [Finetune README](README_finetune.md)

## Change history

* 12/30 (v19) update:
    - support for LoRA network training in kohya_gui.py.
* 12/23 (v18.8) update:
    - Fix for conversion tool issue when the source was an sd1.x diffuser model
    - Other minor code and GUI fix
* 12/22 (v18.7) update:
    - Merge dreambooth and finetune is a common GUI
    - General bug fixes and code improvements
* 12/21 (v18.6.1) update:
    - fix issue with dataset balancing when the number of detected images in the folder is 0

* 12/21 (v18.6) update:
    - add optional GUI authentication support via: `python fine_tune.py --username=<name> --password=<password>`