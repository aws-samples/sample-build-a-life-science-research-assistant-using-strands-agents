# Build a Life Science Research Assistant Using Strands Agents

## Overview

The Drug Discovery Agent is an AI-powered agent designed to assist pharmaceutical researchers in exploring scientific literature, clinical trials, and drug databases. This tool leverages Amazon Bedrock's large language models to provide interactive conversations about drug discovery, target proteins, diseases, and related research.

## Features

- **Interactive Chat Interface**: Engage in natural language conversations about drug discovery topics
- **Multiple Data Sources**: Access information from various scientific databases:
  - arXiv (scientific papers)
  - PubMed (biomedical literature)
  - ChEMBL (bioactive molecules)
  - ClinicalTrials.gov (clinical trials)
  - Web search via Tavily

- **Comprehensive Analysis**: Get detailed information about:
  - Target proteins and their inhibitors
  - Disease mechanisms
  - Drug candidates and their properties
  - Clinical trial results
  - Recent research findings

## Getting Started

### Prerequisites

- Install required Python packages using `pip install -r requirements.txt`.
- Visit [tavily.com](https://tavily.com/) to create a account and generate an API key.
- Obtain [long-term](https://docs.aws.amazon.com/sdkref/latest/guide/access-iam-users.html) or (preferred) [short-term](https://docs.aws.amazon.com/sdkref/latest/guide/access-temp-idc.html) AWS credentials.
- [Request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to the following Amazon Bedrock foundation models in the `us-west-2` region:
  - Anthropic Claude 3.7 Sonnet
  - Anthropic Claude 3.5 Sonnet
  - Anthropic Claude 3.5 Haiku

### Installation

1. Clone this repository
2. Install the required Python dependencies.

   ```sh
   pip install -r requirements.txt
   ```

3. Configure your AWS credentials by [setting them as environment variables](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html), [adding them to a credentials file](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html), or following another [supported process](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html).

4. Save your Tavily API key to a `.env` file in the following format:

   ```sh
   TAVILY_API_KEY="YOUR_API_KEY"
   ```

5. (Optional) Download ttf file for a font you will use and move it to `assets/` and change `font_path` in `chat.py`

### Running the Application

1. Start the Streamlit web interface:

   ```sh
   streamlit run application/app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically <http://localhost:8501>)

## Using the Drug Discovery Agent

1. **Select a Model**: Choose from available foundation models (Claude 3.7 Sonnet, Claude 3.5 Sonnet, or Claude 3.5 Haiku)

2. **Ask Questions**: Examples of questions you can ask:
   - "Please generate a report for HER2 including recent news, recent research, related compounds, and ongoing clinical trials."
   - "Find recent research papers about BRCA1 inhibitors"
   - "What are the most promising drug candidates for targeting coronavirus proteins?"
   - "Summarize the mechanism of action for HER2 targeted therapies"

3. **Generate Reports**: The agent can compile comprehensive reports about specific targets or diseases

## Architecture

The Drug Discovery Agent is built using:

- **Strands Agent SDK**: For creating AI agents with specific capabilities
- **Streamlit**: For the web interface
- **MCP (Model Context Protocol)**: For connecting to external data sources
- **Amazon Bedrock**: For accessing powerful language models like Claude

Each MCP server provides specialized tools for accessing different scientific databases:

- `mcp_server_arxiv.py`: Search and retrieve scientific papers from arXiv
- `mcp_server_chembl.py`: Access chemical and bioactivity data from ChEMBL
- `mcp_server_clinicaltrial.py`: Search and analyze clinical trials
- `mcp_server_pubmed.py`: Access biomedical literature from PubMed
- `mcp_server_tavily.py`: Perform web searches for recent information

## Limitations

- This repository is intended for Proof of Concept (PoC) and demonstration purposes only. It is NOT intended for commercial or production use.
- The agent relies on external APIs which may have rate limits
- Information is limited to what's available in the connected databases

## Future Enhancements

- Integration with additional drug discovery tools and databases
- Enhanced visualization of molecular structures and interactions
- Support for proprietary research databases

## Contributors

- Hasun Yu, Ph.D. (AWS AI/ML Specialist Solutions Architect) | [Mail](mailto:hasunyu@amazon.com) | [LinkedIn](https://www.linkedin.com/in/hasunyu/)

## Citation

- If you find this repository useful, please consider giving a star ⭐ and citation

## References

- [Strands Agents SDK](https://strandsagents.com/0.1.x/)
- [Strands Agents Samples](https://github.com/strands-agents/samples/tree/main)
- [Strands Agents Samples - Korean](https://github.com/kyopark2014/strands-agent)
