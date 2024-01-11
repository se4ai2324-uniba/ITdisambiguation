GitHub Actions Workflow
==============================
In this section, we detail the GitHub Actions workflows employed in our project. GitHub Actions are a powerful tool for automating software workflows, enabling continuous integration and continuous deployment (CI/CD) practices. Our project leverages these workflows to automate various tasks such as code quality checks, automated testing, and deployment processes. Each YAML file described below represents a specific workflow designed for a particular aspect of our software development process. These workflows ensure that our codebase is rigorously tested, adheres to quality standards, and is consistently deployed to our production environment with minimal manual intervention.

## `pylint.yml`
This file configures a GitHub Actions workflow to perform static code analysis using Pylint on Python code. It triggers on pushes to the __stage branch__ and runs on an __ubuntu-latest virtual machine__. The workflow supports Python versions 3.8, 3.9, and 3.10.

The workflow includes the following steps:

- __Code Checkout__: Uses `actions/checkout@v3` to fetch the source code.
- __Python Setup__: Sets up the Python environment using `actions/setup-python@v3`.
- __Dependencies Installation__: Installs Pylint.
- __Code Analysis with Pylint__: Executes Pylint on Python files in the `src` directory, excluding specific errors and limitations, and generates a textual report.
- __Report Upload__: Uploads the Pylint report as a GitHub Actions artifact, available for review and download.
  
This file is crucial for ensuring that the Python code in the repository adheres to the quality and style standards encoded in Pylint's rules.


## `pytest.yml`
This file sets up a GitHub Actions workflow for running automated tests using Pytest on Python code. It is activated by pushes to the __stage branch__, specifically targeting changes in the `src`, tests directories, but excluding the `tests/model_testing` directory.

Key features of this workflow:

- __Trigger Conditions__: It runs when there are changes in the `src` and `tests` directories (excluding `tests/model_testing`) upon push to __stage__.
- __Environment Setup__: The workflow runs on an __ubuntu-latest virtual machine__.
- __Workflow Steps__:
    - __Repository Checkout__: Uses `actions/checkout@v4` for fetching the latest code.
    - __Python Environment Setup__: Utilizes `actions/setup-python@v4` with Python version 3.11.
    - __DVC (Data Version Control) Setup__: Implements `iterative/setup-dvc@v1` for data and model version control.
    - __Installation of Requirements__: Upgrades pip and installs dependencies from requirements.txt.
    - __Data and Model Preparation__: Uses DVC to pull training data and model files, and runs preprocessing steps.
    - __Testing Phases__: Executes multiple Pytest commands to test dataset, preprocessing functions, model behavior, and APIs.
    - __Commented-Out Test for Model Training__: There's a section for model training tests, currently commented out.

This workflow ensures that the codebase remains stable and functional with each new push, covering a wide range of tests from dataset integrity to API functionality.

## `azure-static-web-apps-nice-island-02cd56d03.yml`
This file outlines a CI/CD workflow for Azure Static Web Apps, designed to automate the deployment of a frontend application hosted on Azure. It triggers on push events to the main branch and on specific actions (__opened__, __synchronize__, __reopened__, __closed__) of pull requests, again targeting the __main__ branch.

Key aspects of this workflow:

- __Triggering Conditions__: It's set to run on push events to the __main branch__, specifically for changes in the `frontend/**` directory, and on pull request activities concerning the main branch.
- __Environment and Jobs__:
    - __Build and Deploy Job__:
      - __Condition__: Executes if it's a push event or an open/synchronize/reopen action in a pull request (not on pull request closure).
      - __Platform__: Runs on __ubuntu-latest virtual machine__.
      - __Steps__:
        - __Code Checkout__: Uses `actions/checkout@v3` with submodule and Large File Storage (LFS) settings.
        - __Build and Deploy Action__: Utilizes `Azure/static-web-apps-deploy@v1` for deployment, configured with secrets for Azure and GitHub tokens, and specifies the locations for app source (`/frontend/`), optional API source, and output directory (`dist`).
    - __Close Pull Request Job__:
      - __Condition__: Only runs if the event is a pull request closure.
      - __Platform__: Also runs on __ubuntu-latest virtual machine__.
      - __Steps__:
          - __Close Pull Request Action__: Carries out the closure of the pull request using the same Azure deployment action.

This workflow is crucial for maintaining a streamlined and automated deployment pipeline for the frontend application, ensuring that each update is efficiently built and deployed to Azure Static Web Apps.

## `main_ITdisambiguation.yml`
This YAML file outlines a GitHub Actions workflow for building and deploying a Docker container app to an Azure Web App named ITdisambiguation. The workflow is triggered on pushes to the __main branch__, specifically focusing on changes within the `src/**` directory, and can also be manually triggered via __workflow_dispatch__.

Key elements of this workflow:

- __Trigger Conditions__: Activates on pushes to the main branch (for `src/**` changes) and allows manual triggers.
- __Environment__: Both build and deploy jobs run on __ubuntu-latest virtual machine__.
- __Jobs__:
  - __Build Job__:
    - __Steps__:
      - __Code Checkout__: Uses `actions/checkout@v2`.
      - __Docker Buildx Setup__: Prepares Docker Buildx environment using `docker/setup-buildx-action@v2`.
      - __Docker Registry Login__: Logs in to Docker registry with credentials stored in GitHub secrets.
      - __Python Environment Setup__: Configures Python 3.11 environment.
      - __DVC Setup__: Sets up Data Version Control (DVC) for data and model management.
      - __Data and Model Preparation__: Pulls the model weights using DVC.
      - __Docker Image Build__: Builds the Docker image with the tag based on the commit SHA and pushes it to Docker Hub.
  - __Deploy Job__:
    - __Dependency__: Depends on the successful completion of the build job.
    - __Environment Info__: Specifies production environment and retrieves the web app URL.
    - __Deployment Steps__:
      - __Azure Web App Deployment__: Deploys the Docker image to the Azure Web App (ITdisambiguation) using `azure/webapps-deploy@v2`, with the necessary configuration details and publish profile provided via GitHub secrets.

This workflow plays a critical role in automating the continuous integration and deployment process, ensuring a streamlined deployment of the latest version of the app to Azure Web App.

## Training action
Initially we wanted to make a GitHub Action also to retrain the model whenever code or data changed, but in the end we decided to avoid implementing it.
This choice was due to the fact that GitHub's virtual machines running the actions are not GPU-accelerated, this would make our training so slow that would exceed the maximum job execution time (6 hours).
