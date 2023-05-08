# :parrot: :hugs: :robot: Langchain HuggingGPT

Implementation of [HuggingGPT](https://arxiv.org/abs/2303.17580) with [langchain](https://docs.langchain.com/docs/).

## Getting Started

### Install with pdm

```commandline
pdm install
```

### Install with pip

```commandline
pip install -r requirements.txt
```

## Usage

Run application with pdm:

```commandline
pdm run hugginggpt
```

Or run application directly in your virtual environment of choice:

```commandline
python main.py
```

Then converse with HuggingGPT, e.g:

```
Generate an image of a classroom
```

```
Now count the number of chairs in the image
```

To use the application in standalone mode, use the `--prompt` flag:

```commandline
pdm run hugginggpt --prompt "Generate an image of a classroom"
```

## Credits

* [JARVIS](https://github.com/microsoft/JARVIS)

