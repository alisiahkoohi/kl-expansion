<h1 align="center">Karhunen-Loève expansion</h1>

## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/alisiahkoohi/kl-expansion
cd kl-expansion/
conda env create -f environment.yml
conda activate klexp
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate klexp`, the
following times.

## Usage

To run the example scripts, you can use the following commands.

```bash
python scripts/kl-expansion-toy-example.py --x_range [-10,10] --M 20
```
## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi and Lorenzo Baldassari