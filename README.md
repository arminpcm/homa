# Homa

A repository to train specialized image generation algorithms.

## Project Overview

Homa is designed to simplify the development and training of image generation algorithms. This repository includes tools for linting, testing, and static type checking to ensure high code quality.

---

## Requirements

- Python 3.10.0
- pip (latest version)
- uv
- git

---

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:arminpcm/homa.git
    cd homa
    ```

2. Set up a virtual environment:
    ```bash
    pip install uv
    uv venv .venv --python=/usr/bin/python3.10
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

4. If using `uv` for dependency management, synchronize dependencies:
    ```bash
    uv sync
    ```

---

## Usage

### Linting and Formatting
To lint and format your code as well as perform type checks:
```bash
./homa lint
```

### Running Tests
To run tests using `pytest`:
```bash
./homa test
```

### Starting the Project
To start the Docker Compose environment:
```bash
./homa start
```

To start in detached mode:
```bash
./homa start -d
```

### Stopping the Project
To stop the Docker Compose environment:
```bash
./homa stop
```

### Restarting the Project
To restart the Docker Compose environment:
```bash
./homa restart
```

To restart in detached mode:
```bash
./homa restart -d
```

### Update requirements:
To add new or change existing requirements, modofy [requirements.in](requirements.in) file. Then run the following command to update the [requirements.txt](requirements.txt) file:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip install -r requirements.txt
```

---

## Project Structure

```
homa/
│
├── src/                    # Source code for the project
├── tests/                  # Unit tests for the project
├── README.md               # Project documentation
├── pyproject.toml          # Tool configurations
├── requirements.txt        # Dependencies
├── docker/                 # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
└── .venv/                  # Virtual environment (not included in repo)
```

---

## Contributing

Feel free to open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.