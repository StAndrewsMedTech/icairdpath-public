#!/usr/bin/python3

from click import group, version_option, command, argument
from multiprocessing import  set_start_method
from pathlib import Path

import repath.experiments.wang as wang
import repath.experiments.lee as lee
import repath.experiments.liu as liu
import repath.experiments.tissuedet as tissue
import repath.experiments.cervical_algorithm1 as cervical1
import repath.experiments.cervical_algorithm2 as cervical2
import repath.experiments.cervical_algorithm3 as cervical3
import repath.experiments.cervical_algorithm4 as cervical4
import repath.experiments.cervical_set2_exp1 as cervical_exp1
#import repath.experiments.cervical_set2_exp2 as cervical_exp2
#import repath.experiments.cervical_set2_binary as cervical_bin
import repath.experiments.bloodmuc_rework as bm
import repath.experiments.bloodmuc_sample_size as bm2
import repath.experiments.bloodmuc_sigma as bm3
#import repath.experiments.bloodmuc_lev3 as bm4
#import repath.experiments.bloodmuc_he as bm5
import repath.experiments.bloodmuc_nn as bm6
import repath.experiments.bloodmuc_sigma0 as bm7
import repath.experiments.bloodmuc_sigma0_nn as bm8
import repath.experiments.bloodmuc_sigma0_nn_sampsize as bm9
import repath.experiments.bloodmuc_nn_subexp1 as bm10
import repath.experiments.bloodmuc_nn_subexp2 as bm11
import repath.experiments.bloodmuc_nn_subexp3 as bm12
import repath.experiments.endo_set1 as endo1
import repath.experiments.endo_set2 as endo2
import repath.experiments.endo_512 as endo512
import repath.experiments.endo_1024 as endo1024
import repath.experiments.endo_1024_bmmode as endo1024bm
import repath.experiments.endo_512_bmmode as endo512bm
#import repath.experiments.endo_512_bmpatch as endo512bp
import repath.experiments.cervical_full_set as cervical_full_set
#import repath.experiments.endo_set1 as endo1
import repath.experiments.endo_set2 as endo2
import repath.experiments.cervical_1024_full as cervical1024
import repath.experiments.cervical_512_full as cervical512
import repath.experiments.cervical_256 as cervical256
import repath.experiments.cervical_1024 as cervical1024
import repath.experiments.cervical_256_subCat as cervicalsubCat
<<<<<<< HEAD
import repath.experiments.cervical_256_balanced as cervical256balanced
=======
from repath.final_algo.single_slide import single_slide_prediction
>>>>>>> fed9814d89c7a8d1f3ac5bf84f7d7fba1b3877c1

@group()
@version_option("1.0.0")
def main():
    pass


@command()
@argument("experiment")
@argument("step", required=False)
def run(experiment: str, step: str = None) -> None:
    """Run an EXPERIMENT with optional STEP."""
    print(f"{experiment}: {step}")
    eval(f"{experiment}.{step}()")


@command()
@argument("experiment", required=False)
def show(experiment: str) -> None:
    """List all the experiments or all the steps for an EXPERIMENT."""
    print(f"{experiment}")


@command()
@argument("input_slide", required=True)
@argument("device_idx", required=False, default=0)
def cervical(input_slide: Path, device_idx: int) -> None:
    """Run final cervical screen algorithm"""
    single_slide_prediction(input_slide, 'cerv', device_idx)


@command()
@argument("input_slide", required=True)
@argument("device_idx", required=False, default=0)
def endometrial(input_slide: Path, device_idx: int) -> None:
    """Run final cervical screen algorithm"""
    single_slide_prediction(input_slide, 'endo', device_idx)


main.add_command(run)
main.add_command(show)
main.add_command(cervical)
main.add_command(endometrial)


if __name__ == "__main__":
    set_start_method('spawn')
    main()
    
