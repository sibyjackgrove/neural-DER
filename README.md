
**Status:** Expect regular updates and bug fixes.
# Utilitiy for training neural network models for emulating PV-DER dynamics

Solar photovoltaic distributed energy resources (PV-DER) are power electronic inverter based generation (IBG) connected to the electric power distribution system (eg. roof top solar PV systems). This utility can be used to simulate the behaviour a single DER connected to a stiff voltage source as shown in the following schematic:

![schematic of PV-DER](PVDER_schematic.png)

## Basics


## Links
* Source code repository:

## Installation
You can install the module directly from github with following commands:
```
git clone 
cd neural-DER
pip install -e .
```
## Using the module
The module can be imported as a normal python module:
```
import neuralder
```

Dependencies: SciPy, Numpy, Matlplotlib

## Issues
Please feel free to raise an issue when bugs are encountered or if you are need further documentation.

## Who is responsible?
- Siby Jose Plathottam sibyjackgrove@gmail.com

## Citation
If you use this code please cite it as:
```
@misc{neuralder,
  title = {{neural-DER}: A utility training and evaluating neural network models for PV-DER dynamics},
  author = "{Siby Jose Plathottam}",
  howpublished = {\url{}},
  url = "",
  year = 2019,
  note = "[Online; accessed ]"
}
```
### Copyright and License
Copyright © 2019, Plathottam, Siby Jose

neural-DER is distributed under the terms of [BSD-3 OSS License.](LICENSE)
