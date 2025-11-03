# Quality Metrics for AI-Generated Asset Administration Shells: A Perturbation-Based Evaluation Approach

This project uses **Docker** and **VS Code Dev Containers** to create a OS-independent reproducible development environment to reproduce the experiments conducted for the publication `Quality Metrics for AI-Generated Asset Administration Shells: A Perturbation-Based Evaluation Approach`. At the moment project is structured using five notebooks to produce the published results. In the future however, the authors plan to integrate the evaluation procedure into a more easily accesible evaluation framework. This README will guide you through the setup process and project structure. If you have trouble reproducing the results please contact the authors janek.gross@hs-mainz.de and elena.zentgraf@hs-mainz.de.

---

## Prerequisites

Before you start working with this project, make sure you have the following installed:

1. **Docker**:
   Docker is required to run the containers that provide the development environment. Install Docker from [Docker's official website](https://www.docker.com/get-started).

   After installation, make sure Docker is running by executing the following command:

   ```bash
   docker --version
   ```

   If the command returns the version number of Docker, you're good to go!

2. **VS Code**:
   Visual Studio Code (VS Code) is the primary IDE used for development. You can download it from [VS Code's official website](https://code.visualstudio.com/).

3. **VS Code Extensions**:
   - **Remote - Containers**: This extension allows you to work inside a Docker container directly from VS Code.
   - **Python**: If you're working with Python, this extension is recommended for syntax highlighting and IntelliSense.
   - **Jupyter**: This extension is useful if you are working with Jupyter Notebooks.

---

## Project Structure

After buidling the container, inside you see:

```
data/ (mounted data directory to share data with host and all services)
├── raw/sample           (unprocessed raw data (AASX files and PDF datasheets))
└── processed/sample     (processing directory)
perturbation_experiments/ (this repository mounted inside the container)
├── devcontainer-perturbation_experiments/
│   └── .devcontainer/
│       └── devcontainer.json  (for experiments service)
├── notebooks/ (experiment code and scripts)
├── pdftoaas/ (external tool in the exact version that was used in this project including the modified dockerfile)
├── .env.example (example configuration for environment variables)
├── docker-compose.yml  (to configuration docker service)
├── README.md

```

### Key Components:
- **`devcontainer-perturbation_experiments/`**: This directory contains the configuration for the development container that attaches to the [basyx-pdf-to-aas](https://github.com/eclipse-basyx/basyx-pdf-to-aas) tool and installs the requirements for the metrics evaluation
- **`.devcontainer/`**: The project's service has its own `.devcontainer` folder containing the `devcontainer.json` file, which specifies how VS Code should build and open the service container. This configuration file tells VS Code how to access or create a development container with a well-defined tool and runtime stack. 
- **`docker-compose.yml`**: This file defines the Docker containere for the `experiments` service and it's dependencies. It specifies how the container should be built, which ports it uses and which directory is mounted.

---

## Setting Up the Project

### 1. **Clone the Project**:

Clone the project from the repository (if you haven’t already):

```bash
git clone --recurse-submodules <repository-url> (or use git submodule update --init --recursive)
cd <project-folder>
```
### 2. **Setting up the Environment**:

Create a copy of **.env.example** and name it **.env**.
Fill in your API key and the absolute path to your data.

### 3. **Open VS Code**:

Open VS Code in the desired workspace folder devcontainer-perturbation_experiments/:

```bash
cd <workspace-folder>
code .
```

VS Code will automatically detect the `devcontainer.json` files in the `devcontainer-evaluation/` or `devcontainer-pdftoaas/` directories and prompt you to reopen the folder in a Docker container.

---

### 4. **Rebuild or Restart the Container**:

If you need to rebuild the container for any reason (e.g., after changing the Docker configuration), open the Command Palette in VS Code (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) and search for:

- **Dev Containers: Rebuild Container**
- **Dev Containers: Reopen in Container**

This will rebuild and restart the container with the latest configurations.


---


## Troubleshooting

### Common Issues:
1. **VS Code not reopening in container**: If VS Code doesn’t reopen in the container, ensure you have the **Dev Containers** extension installed. You can also manually open the Command Palette and select **Reopen in Container**.

2. **Containers not starting correctly**: Run `docker-compose logs` to check the logs for any errors related to container startup. You can also try rebuilding the containers using `docker-compose build`.

3. **Environment not set correctly**: Double-check that the correct Python environment or other configurations are set in the `devcontainer.json` for each service.

4. **VS-Code doesn't recognize the jupyter kernel**: Select the Python 3.12.9 /usr/local/bin/python environment. If it doesn't work or the kernel gets stuck try ctrl+shif+P `reload window`.

---

## License
This repository is licensed under the MIT License (see [LICENSE](./LICENSE)).  
The `pdftoaas` subfolder includes code from the [pdftoaas project], which retains its original MIT License (see [`pdftoaas/LICENSE`](./pdftoaas/LICENSE)).
