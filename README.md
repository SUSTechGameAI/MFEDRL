# Multifaceted Experience-Driven PCG via Reinforcement Learning (MFEDRL)

This is the code for the paper "The Fun Facets of Mario: Multifaceted Experience-Driven PCG via Reinforcement Learning" accepted by the 13th Workshop on Procedural Content Generation.

Please use this bibtex if you use this repository in your work:

````
@inproceedings{wang2022mfedrl,
  title={The Fun Facets of Mario: Multifaceted Experience-Driven PCG via Reinforcement Learning},
  author={Wang, Ziqi and Liu, Jialin and Yannakakis, Georgios N},
  booktitle = {13th Workshop on Procedural Content Generation at the 2022 International Conference on the Foundations of Digital Games},
  year={2022},
  pages={Accepted},
  organization={ACM}
}
````
### Environments that have been tested
* Python 3.9.6
* JPype 1.3.0
* pygame 2.0.1
* dtw 1.4.0
* scipy 1.7.2
* torch 1.9.0+cu111
* numpy 1.20.3
* gym 0.21.0
### How to use
#### Training GAN generator:
Run command line instruction:
````
At the root path of this project> python train.py generator
````
You can check the running arguments (to specify algorithm parameters) by:
````
At the root path of this project> python train.py generator --help
````
#### Training CNet:
Run command line instruction:
````
At the root path of this project> python train.py cnet
````
You can check the running arguments (to specify algorithm parameters) by:
````
At the root path of this project> python train.py cnet --help
````
#### Training CNet:
Run command line instruction:
````
At the root path of this project> python train.py designer
````
You can check the running arguments (to specify algorithm parameters) by:
````
At the root path of this project> python train.py designer --help
````
#### Play a level:
You may modify the path of level file in line 322 of the _smb.py_, and then run _smb.py_ to play any level stored in a Mario-AI-Framework-supported text file.
* Line 322 of _smb.py_
````
    lvl = MarioLevel.from_file(Path of your level file)
````
The path can be either related (to project root) or absolute, and the file type won't be checked.