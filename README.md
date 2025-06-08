# AshaVoice: Local LLM-Powered Medical Scribe Assistant

AshaVoice is a proof-of-concept project demonstrating a voice-to-text pipeline integrated with a local Large Language Model (LLM) to assist in generating medical documentation. It transcribes patient-doctor conversations, summarizes symptoms, and generates structured SOAP (Subjective, Objective, Assessment, Plan) notes, all running entirely on your local machine using open-source models.

---

🚀 **Features**

- 🎙️ **Local Speech-to-Text (STT):** Utilizes openai-whisper for accurate audio transcription, running on your CPU or GPU.
- 🧠 **Local Large Language Model (LLM):** Integrates with Ollama to host powerful LLMs (like Llama3) locally, ensuring data privacy and reducing API costs.
- 📝 **Structured Medical Documentation:** Generates patient symptom summaries and SOAP notes in a structured JSON format.
- 🛠️ **Robust Development Environment:** Leverages Docker, Docker Compose, and VSCode DevContainers for consistent and reproducible development.
- ✨ **Code Quality & Automation:** Incorporates Ruff for linting and formatting, and pre-commit hooks for automatic code checks.
- ⚡ **Task Automation:** Uses Task (Taskfile.dev) for streamlined command execution for both local host and DevContainer environments.

---

🏗️ **Project Structure**

```txt
AshaVoice/
├── .devcontainer/                # VSCode DevContainer configuration
│   ├── devcontainer.json         # DevContainer setup (Python, extensions, post-create commands)
│   └── docker-compose.yml        # Docker Compose for DevContainer services (backend, ollama)
├── .git/                         # Git version control directory
├── .github/                      # CI/CD workflows (future)
├── .gitignore                    # Specifies intentionally untracked files to ignore
├── .pre-commit-config.yaml       # Pre-commit hooks configuration
├── Taskfile.yaml                 # Task runner commands for host machine
├── Taskfile.devcontainer.yaml    # Task runner commands for inside DevContainer
├── README.md                     # Project README (you are here!)
├── backend/                      # Backend Python application
│   ├── app/                      # Main application code
│   │   ├── api/                  # API definitions (e.g., Pydantic models)
│   │   │   └── v1/
│   │   │       └── models.py     # Pydantic models for structured output
│   │   └── core/                 # Core logic for STT and LLM agent
│   │       ├── agent.py          # LLM agent logic (Ollama integration, prompt engineering)
│   │       └── stt.py            # Speech-to-Text integration (Whisper)
│   ├── Dockerfile                # Dockerfile for building the backend service
│   ├── pipeline.py               # Core pipeline execution logic
│   ├── Pipfile                   # Pipenv dependency definitions
│   ├── Pipfile.lock              # Pipenv locked dependencies for reproducible builds
│   ├── pyproject.toml            # Ruff linter/formatter configuration
│   └── tests/                    # Unit tests (placeholder for now)
├── data/                         # Input/Output data (audio files, generated notes)
│   ├── output/                   # Directory for generated summary/SOAP notes
│   └── patient_intro.mp3         # Example audio input file
├── docker-compose.test.yml       # Docker Compose for standalone local testing (no DevContainer)
├── models/                       # Persistent storage for downloaded LLM/STT models
│   └── whisper_cache/            # Whisper model weights will be downloaded here
└── scripts/                      # Utility scripts
    └── run_pipeline.sh           # Bash script to orchestrate the pipeline from the root
```

---

⚙️ **Prerequisites**

Before you begin, ensure you have the following installed on your host machine:

- **Git:** For cloning the repository.
- **Docker Desktop:** Includes Docker Engine and Docker Compose.
- **VSCode (Optional but Recommended):** For using DevContainers.
- **Task (go-task):** A command runner to simplify common operations.
  Installation: `brew install go-task/tap/go-task` (macOS), or see [Task Installation Guide](https://taskfile.dev/#/installation).
- **Python 3.12 and pipenv:** If you plan to run commands outside the DevContainer or manage dependencies manually on your host for testing.
- **pre-commit:** Install globally or in a base environment on your host: `pip install pre-commit`.

---

🚀 **Getting Started**

1. **Clone the Repository**

   ```bash
   git clone <https://github.com/your-username/AshaVoice.git>
   cd AshaVoice
   ```

2. **Prepare Example Data**

   Place a test audio file (e.g., `patient_intro.mp3`) in the `data/` directory. You can use any `.mp3` or `.wav` file containing spoken English.

   Example:

   ```bash
   # Create the data directory
   mkdir -p data

   # (Manually place your audio file here, e.g., patient_intro.mp3)
   ```

3. **Choose Your Development Environment**

   - **Option A:** VSCode DevContainers (Recommended for Development)
     This is the easiest way to get a consistent and isolated development environment.
     - Open in VSCode: Open the AshaVoice folder in VSCode.
     - Reopen in Container: VSCode should prompt you to "Reopen in Container". If not, open the Command Palette (`F1` or `Ctrl+Shift+P`) and select "Remote-Containers: Reopen in Container".
     - Initial Setup: The DevContainer will build the Docker images, install Python dependencies (pipenv), and pull the llama3 LLM model for Ollama. This first time will take a while (especially downloading llama3).
     - Verify: Once the container is ready, open a terminal in VSCode (`/workspaces/AshaVoice`). You should see both backend and ollama services running (`docker ps`). The llama3 model should be pulled.

   - **Option B:** Standalone Docker Compose (For Local Testing/CI)
     If you prefer to run services without VSCode DevContainers (e.g., for local testing or CI/CD pipelines):
     - Start Services:

       ```bash
       task test-local-up
       ```

       This will build the backend image, start the ollama and backend containers, and pull the llama3 model.
     - Stop Services:

       ```bash
       task test-local-down
       ```

---

🏃 **Running the Pipeline**

Once your environment is set up (either DevContainer or `test-local-up`):

1. **Run Pipeline from Project Root (Recommended for Development)**
   This uses the `scripts/run_pipeline.sh` script to orchestrate the process within your DevContainer or locally-running backend service.

   ```bash
   task run-pipeline -- data/patient_intro.mp3
   ```

   Replace `data/patient_intro.mp3` with the actual path to your audio file relative to the project root.
   You'll see detailed log messages in your terminal, and the generated summary, SOAP note, and transcript will be saved to `data/output/`.

2. **Run Pipeline Inside DevContainer (Manual Execution)**
   If you're in the DevContainer terminal and want to run the Python script directly:

   ```bash
   cd backend
   pipenv run python pipeline.py ../data/patient_intro.mp3
   ```

3. **Run Pipeline in Standalone Test Environment (Manual Execution)**
   If you used `task test-local-up` and want to manually execute inside the backend container:

   ```bash
   task test-local-exec -- pipenv run python pipeline.py ../data/patient_intro.mp3
   ```

---

🧹 **Code Quality & Automation**

- **Linting and Formatting (Ruff)**
  We use Ruff for fast linting and formatting.
  Run from inside DevContainer:

  ```bash
  task -f Taskfile.devcontainer.yaml lint
  ```

  This will automatically fix most issues based on the rules in `backend/pyproject.toml`.

- **Pre-commit Hooks**
  Pre-commit automatically runs checks (like Ruff, YAML syntax, trailing whitespace) before you commit your changes to Git.
  - Install hooks (first time only on your host machine):

    ```bash
    task pre-commit-install
    ```

  - Update hooks:

    ```bash
    task pre-commit-update
    ```

  Now, every git commit will automatically run these checks.

---

🧪 **Testing**

- **Python Unit Tests**
  *(Coming Soon)*
  You can add your Python unit tests in the `backend/tests/` directory.
  Run from inside DevContainer:

  ```bash
  task -f Taskfile.devcontainer.yaml test-python
  ```

---

💡 **Notes on Local Models**

- **Whisper Models:** The base Whisper model weights (approx. 70-80 MB) are downloaded into `models/whisper_cache/` when `transcribe_audio` is first run. This directory is mounted as a Docker volume, so the models persist across container restarts.
- **Ollama Models:** The llama3 model (approx. 4.7 GB) is pulled by the ollama service and stored in a Docker volume (`ollama_data`). This ensures it persists and doesn't need to be re-downloaded.

---

🤝 **Contributing**

Contributions are welcome! Please feel free to open issues or submit pull requests.

---

📄 **License**

This project is open-source and available under the MIT License.
