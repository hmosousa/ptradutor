# PTradutor

## Getting Started

This guide will help you set up a development environment, download the news data, and translate it into the language of your choice.

### Setting Up Your Development Environment

To begin, you'll need to create a virtual environment and install the necessary dependencies. Follow these steps:

```sh
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the project dependencies
pip install -e .
```

### Translating 

```sh
python scripts/translate.py -l <target_language> -n <dataset> -d <domain> -s <split>
```

Please replace `<target_language>` with the code of the language you want to translate the texts into. Run `list_languages` in the terminal to check the available languages.


## License

The code of this project is released under the MIT License.

## Contact

For questions, suggestions, or collaborations, please contact the project maintainer:

- **Project Maintainer:** u1537782 
- **Email:** u1537782.o.@inesctec.pt

## Citation

If you use this work, please cite:

```bibtex
@article{Sousa_Almasian_Campos_Jorge_2025,
  title     = {Tradutor: Building a Variety Specific Translation Model},
  author    = {Sousa, Hugo and Almasian, Satya and Campos, Ricardo and Jorge, Alipio},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {24},
  pages     = {25183--25191},
  year      = {2025},
  month     = {Apr},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/34704},
  doi       = {10.1609/aaai.v39i24.34704},
```
