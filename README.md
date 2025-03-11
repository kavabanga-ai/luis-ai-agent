# Liver: A LangGraph AI-Agent

## Overview

AI-Agent is an intelligent agent framework built using [LangGraph](https://smith.langchain.com/). It leverages LangGraph's capabilities to visualize, debug, and interact with complex agent-based workflows.

## Prerequisites

- **Python 3.11+**
- **LangGraph CLI**
- **LangGraph Platform**
- **LangGraph Server**
- **LangSmith Account**
- **Docker (optional, for deployment)**

## Installation

### 1. Install Python and Dependencies
```sh
# Ensure Python 3.11+ is installed
python --version

# Install virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install LangGraph CLI and Required Packages
```sh
# Install LangGraph CLI
pip install "langgraph-cli[inmem]"

# Verify installation
langgraph --version

```

## Running the Application

### Local Development Server
```sh
langgraph dev
```
By default, the server runs at `http://127.0.0.1:2024` and opens LangGraph Studio. If not, access it manually:
[Open LangGraph Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)

### Deployed Application
1. Deploy via LangGraph CLI:
   ```sh
   langgraph up
   ```
2. Access deployment in LangSmith UI.
3. Click **LangGraph Studio** to interact with AI-Agent.

## Troubleshooting

- Ensure environment variables are set correctly.
- Verify `langgraph.json` is properly formatted.
- Check **LangSmith data region** settings if errors occur.

