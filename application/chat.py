# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import asyncio
import datetime
import logging
import os
import sys
import traceback
import uuid

import info
from botocore.config import Config
from mcp import StdioServerParameters, stdio_client
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from strands_tools import file_write

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("chat")

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models = info.get_model_info(model_name)
reasoning_mode = "Disable"
os.environ["BYPASS_TOOL_CONSENT"] = "true"  # Bypass consent for file_write


def update(modelName, reasoningMode):
    global model_name, model_id, model_type, reasoning_mode

    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")

        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

    if reasoningMode != reasoning_mode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")


def initiate():
    global userId
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")


#########################################################
# Strands Agent Model Configuration
#########################################################
def get_model():
    profile = models[0]
    if profile["model_type"] == "nova":
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile["model_type"] == "claude":
        STOP_SEQUENCE = "\n\nHuman:"

    if model_type == "claude":
        maxOutputTokens = 64000  # 4k
    else:
        maxOutputTokens = 5120  # 5k

    maxReasoningOutputTokens = 64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens - 1000)

    if reasoning_mode == "Enable":
        model = BedrockModel(
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=64000,
            stop_sequences=[STOP_SEQUENCE],
            temperature=1,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            },
        )
    else:
        model = BedrockModel(
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=maxOutputTokens,
            stop_sequences=[STOP_SEQUENCE],
            temperature=0.1,
            top_p=0.9,
            additional_request_fields={"thinking": {"type": "disabled"}},
        )
    return model


conversation_manager = SlidingWindowConversationManager(
    window_size=10,
)

# MCP Clients for various scientific databases
tavily_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="python", args=["application/mcp_server_tavily.py"]
        )
    )
)

arxiv_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="python", args=["application/mcp_server_arxiv.py"]
        )
    )
)

pubmed_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="python", args=["application/mcp_server_pubmed.py"]
        )
    )
)

chembl_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="python", args=["application/mcp_server_chembl.py"]
        )
    )
)

clinicaltrials_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="python", args=["application/mcp_server_clinicaltrial.py"]
        )
    )
)

#########################################################
# Specialized Tool Agents
#########################################################


@tool
def planning_agent(query: str) -> str:
    """
    A specialized planning agent that analyzes the research query and determines
    which tools and databases should be used for the investigation.

    Args:
        query: The research question about drug discovery or target proteins

    Returns:
        A structured plan outlining tools to use and search queries for each database
    """
    try:
        # Create a planning specialist agent
        planning_system = """
        You are a specialized planning agent for drug discovery research. Your role is to:
        
        1. Analyze research questions to identify target proteins, compounds, or biological mechanisms
        2. Determine which databases would be most relevant (Arxiv, PubMed, ChEMBL, ClinicalTrials.gov)
        3. Generate specific search queries for each relevant database
        4. Create a structured research plan
        
        Return a JSON-formatted plan with:
        1. The overall research approach
        2. For each relevant database: 
           - The specific search queries to use
           - Expected information to extract
        """

        model = get_model()
        planner = Agent(
            model=model,
            system_prompt=planning_system,
        )

        # Ask the planner to create a research plan
        planning_prompt = f"""
        Create a detailed research plan for this drug discovery query: "{query}"
        
        Your plan should specify:
        1. Which databases to search (Arxiv, PubMed, ChEMBL, ClinicalTrials.gov)
        2. Specific search queries for each database
        3. What information to extract from each search
        
        Format your response as a structured plan that can be followed step-by-step.
        """

        response = planner(planning_prompt)
        return str(response)
    except Exception as e:
        logger.error(f"Error in planning agent: {e}")
        return f"Error in planning agent: {str(e)}"


def web_search_agent_impl(
    query: str, active_client, search_type: str = "general"
) -> str:
    """
    Specialized agent for searching the web using Tavily's search engine.

    Args:
        query: The search query
        active_client: Active MCP client session (required)
        search_type: Type of search to perform - "general", "answer", or "news" (default: "general")

    Returns:
        Structured information from web search results
    """
    # Validate that active_client is provided and valid
    if active_client is None:
        error_msg = "Error: Active Tavily client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # Validate client session is usable
        tavily_tools = active_client.list_tools_sync()
        if not tavily_tools:
            error_msg = (
                "Error: Tavily client session is invalid or has no available tools"
            )
            logger.error(error_msg)
            return error_msg

        logger.info(f"tavily_tools: {tavily_tools}")

        # Create a specialized web search agent
        system_prompt = """
        You are a specialized web search agent. Your role is to:
        1. Analyze the query to determine the best search strategy
        2. Search the web using Tavily's search tools
        3. For general searches: Return comprehensive, well-formatted results from across the web
        4. For answer searches: Return a direct answer with supporting evidence
        5. For news searches: Return recent news articles relevant to the query
        6. Always include source URLs for verification
        7. Summarize and highlight key information from search results
        """

        model = get_model()

        # Create the agent with the provided client tools
        web_agent = Agent(model=model, system_prompt=system_prompt, tools=tavily_tools)

        # Build an enhanced query based on search type
        enhanced_query = query
        if search_type.lower() == "news":
            enhanced_query = f"RECENT NEWS: {query}"
        elif search_type.lower() == "answer":
            enhanced_query = f"DIRECT ANSWER: {query}"

        # Execute the search
        response = web_agent(enhanced_query)

        return str(response)
    except Exception as e:
        error_msg = f"Error in web search agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


def arxiv_research_agent_impl(query: str, active_client) -> str:
    """
    Specialized agent for searching Arxiv database for scientific papers.

    Args:
        query: The search query for Arxiv
        active_client: Active MCP client session (required)

    Returns:
        Summarized findings from Arxiv papers
    """
    # Validate that active_client is provided and valid
    if active_client is None:
        error_msg = "Error: Active Arxiv client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # Validate client session is usable
        arxiv_tools = active_client.list_tools_sync()
        if not arxiv_tools:
            error_msg = (
                "Error: Arxiv client session is invalid or has no available tools"
            )
            logger.error(error_msg)
            return error_msg

        logger.info(f"arxiv_tools: {arxiv_tools}")

        # Create a specialized Arxiv research agent
        system_prompt = """
        You are a specialized Arxiv research agent. Your role is to:
        1. Search Arxiv for scientific papers related to the query
        2. Extract and summarize the most relevant findings
        3. Identify key researchers and methodologies
        4. Return structured, well-cited information
        """

        model = get_model()

        arxiv_agent = Agent(model=model, system_prompt=system_prompt, tools=arxiv_tools)

        response = arxiv_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in arxiv research agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


def pubmed_research_agent_impl(query: str, active_client) -> str:
    """
    Specialized agent for searching PubMed database for medical papers.

    Args:
        query: The search query for PubMed
        active_client: Active MCP client session (required)

    Returns:
        Summarized findings from PubMed papers
    """
    # Validate that active_client is provided and valid
    if active_client is None:
        error_msg = "Error: Active PubMed client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # Validate client session is usable
        pubmed_tools = active_client.list_tools_sync()
        if not pubmed_tools:
            error_msg = (
                "Error: PubMed client session is invalid or has no available tools"
            )
            logger.error(error_msg)
            return error_msg

        logger.info(f"pubmed_tools: {pubmed_tools}")

        # Create a specialized PubMed research agent
        system_prompt = """
        You are a specialized PubMed research agent. Your role is to:
        1. Search PubMed for medical papers related to the query
        2. Extract and summarize the most relevant clinical findings
        3. Identify key research groups and methodologies
        4. Return structured, well-cited information with PMID references
        """

        model = get_model()

        pubmed_agent = Agent(
            model=model, system_prompt=system_prompt, tools=pubmed_tools
        )

        response = pubmed_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in pubmed research agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


def chembl_research_agent_impl(query: str, active_client) -> str:
    """
    Specialized agent for searching ChEMBL database for compound information.

    Args:
        query: The search query for ChEMBL
        active_client: Active MCP client session (required)

    Returns:
        Structured information about compounds, targets, and bioactivity
    """
    # Validate that active_client is provided and valid
    if active_client is None:
        error_msg = "Error: Active ChEMBL client session is required but not provided"
        logger.error(error_msg)
        return error_msg

    try:
        # Validate client session is usable
        chembl_tools = active_client.list_tools_sync()
        if not chembl_tools:
            error_msg = (
                "Error: ChEMBL client session is invalid or has no available tools"
            )
            logger.error(error_msg)
            return error_msg

        logger.info(f"chembl_tools: {chembl_tools}")

        # Create a specialized ChEMBL research agent
        system_prompt = """
        You are a specialized ChEMBL research agent. Your role is to:
        1. Extract either the compound name or target name from the query
        2. Search ChEMBL with the name
        3. Return structured, well-formatted compound information with SMILES and activity information for the name
        """

        model = get_model()

        chembl_agent = Agent(
            model=model, system_prompt=system_prompt, tools=chembl_tools
        )

        response = chembl_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in chembl research agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


def clinicaltrials_research_agent_impl(query: str, active_client) -> str:
    """
    Specialized agent for searching ClinicalTrials.gov database.

    Args:
        query: The search query for ClinicalTrials.gov
        active_client: Active MCP client session (required)

    Returns:
        Information about relevant clinical trials
    """
    # Validate that active_client is provided and valid
    if active_client is None:
        error_msg = (
            "Error: Active ClinicalTrials client session is required but not provided"
        )
        logger.error(error_msg)
        return error_msg

    try:
        # Validate client session is usable
        clinicaltrials_tools = active_client.list_tools_sync()
        if not clinicaltrials_tools:
            error_msg = "Error: ClinicalTrials client session is invalid or has no available tools"
            logger.error(error_msg)
            return error_msg

        logger.info(f"clinicaltrials_tools: {clinicaltrials_tools}")

        # Create a specialized ClinicalTrials research agent
        system_prompt = """
        You are a specialized ClinicalTrials.gov research agent. Your role is to:
        1. Search ClinicalTrials.gov for trials related to the query
        2. Extract information about trial status, phases, and results
        3. Identify key sponsors and research groups
        4. Return structured, well-formatted trial information with NCT identifiers
        """

        model = get_model()

        clinicaltrials_agent = Agent(
            model=model, system_prompt=system_prompt, tools=clinicaltrials_tools
        )

        response = clinicaltrials_agent(query)
        return str(response)
    except Exception as e:
        error_msg = f"Error in clinicaltrials research agent: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def synthesis_agent(research_results: str) -> str:
    """
    Specialized agent for synthesizing research findings into a comprehensive report.

    Args:
        research_results: Combined results from all research agents

    Returns:
        A comprehensive, structured scientific report
    """
    try:
        # Create a synthesis agent
        system_prompt = """
        You are a specialized synthesis agent for drug discovery research. Your role is to:
        
        1. Integrate findings from multiple research databases (Arxiv, PubMed, ChEMBL, ClinicalTrials)
        2. Create a comprehensive, coherent scientific report
        3. Highlight key insights, connections, and opportunities
        4. Organize information in a structured, accessible format
        5. Include proper citations and references
        
        Your reports should follow this structure:
        1. Executive Summary (300 words)
        2. Target Overview (biological function, structure, disease mechanisms)
        3. Research Landscape (latest findings and research directions)
        4. Drug Development Status (known compounds, clinical trials)
        5. References (comprehensive listing of all sources)
        """

        model = get_model()
        synthesis = Agent(
            model=model,
            system_prompt=system_prompt,
        )

        # Ask synthesis agent to create a report
        synthesis_prompt = f"""
        Create a comprehensive scientific report based on the following research findings:
        
        {research_results}
        
        Follow the required report structure:
        1. Executive Summary (300 words)
        2. Target Overview
        3. Research Landscape
        4. Drug Development Status
        5. References
        """

        response = synthesis(synthesis_prompt)
        return str(response)
    except Exception as e:
        logger.error(f"Error in synthesis agent: {e}")
        return f"Error in synthesis agent: {str(e)}"


@tool
def generate_pdf_report(report_content: str, filename: str) -> str:
    try:
        # Ensure directory exists
        os.makedirs("reports", exist_ok=True)

        # Add timestamp to filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = f"{filename}_{timestamp}"

        # Set up the PDF file
        filepath = f"reports/{timestamped_filename}.pdf"
        doc = SimpleDocTemplate(filepath, pagesize=letter)

        font_path = "assets/AmazonEmber_Lt.ttf"
        pdfmetrics.registerFont(TTFont("AmazonEmber", font_path))

        # Create styles
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(name="Normal_KO", fontName="AmazonEmber", fontSize=10)
        )
        styles.add(
            ParagraphStyle(name="Heading1_KO", fontName="AmazonEmber", fontSize=16)
        )

        # Process content
        elements = []
        lines = report_content.split("\n")

        for line in lines:
            if line.startswith("# "):
                elements.append(Paragraph(line[2:], styles["Heading1_KO"]))
                elements.append(Spacer(1, 12))
            elif line.startswith("## "):
                elements.append(Paragraph(line[3:], styles["Heading2"]))
                elements.append(Spacer(1, 10))
            elif line.startswith("### "):
                elements.append(Paragraph(line[4:], styles["Heading3"]))
                elements.append(Spacer(1, 8))
            elif line.strip():  # Skip empty lines
                elements.append(Paragraph(line, styles["Normal_KO"]))
                elements.append(Spacer(1, 6))

        # Build PDF
        doc.build(elements)

        return f"PDF report generated successfully: {filepath}"
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")

        # Fallback to text file
        try:
            # Use the same timestamped filename for text fallback
            text_filepath = f"reports/{timestamped_filename}.txt"
            with open(text_filepath, "w", encoding="utf-8") as f:
                f.write(report_content)
            return f"PDF generation failed. Saved as text file instead: {text_filepath}"
        except Exception as text_error:
            return f"Error generating report: {str(e)}. Text fallback also failed: {str(text_error)}"


#########################################################
# MCP Client Session Distribution Mechanism
#########################################################


class MCPClientSessionManager:
    """Manages and distributes MCP client sessions to specialized agent tools"""

    def __init__(self):
        self._active_clients = {}
        self._session_status = {}

    def set_active_clients(self, client_sessions: dict):
        """
        Set the active MCP client sessions for distribution to agent tools

        Args:
            client_sessions: Dictionary mapping client types to active MCP client instances
        """
        self._active_clients = client_sessions.copy()
        # Track session status for each client
        for client_type, client in client_sessions.items():
            self._session_status[client_type] = {
                "active": True,
                "client": client,
                "last_used": None,
            }
        logger.info(
            f"Active MCP client sessions set: {list(self._active_clients.keys())}"
        )

    def get_client(self, client_type: str):
        """
        Get an active MCP client session by type

        Args:
            client_type: Type of client ('tavily', 'arxiv', 'pubmed', 'chembl', 'clinicaltrials')

        Returns:
            Active MCP client instance or None if not available
        """
        if client_type in self._active_clients:
            client = self._active_clients[client_type]
            # Update last used timestamp
            import datetime

            self._session_status[client_type]["last_used"] = datetime.datetime.now()
            return client
        return None

    def get_all_clients(self) -> dict:
        """Return dictionary of all active MCP client sessions"""
        return self._active_clients.copy()

    def is_client_available(self, client_type: str) -> bool:
        """Check if a specific client type is available and active"""
        return client_type in self._active_clients and self._session_status.get(
            client_type, {}
        ).get("active", False)

    def get_session_status(self) -> dict:
        """Get status information for all client sessions"""
        return self._session_status.copy()


# Global session manager instance
_session_manager = MCPClientSessionManager()

#########################################################
# Agent Tool Wrappers for Orchestrator
#########################################################


@tool
def web_search_agent(query: str, search_type: str = "general") -> str:
    """
    Specialized agent for searching the web using Tavily's search engine.

    Args:
        query: The search query
        search_type: Type of search to perform - "general", "answer", or "news" (default: "general")

    Returns:
        Structured information from web search results
    """
    client = _session_manager.get_client("tavily")
    if client is None:
        return "Error: Tavily client session not available"
    return web_search_agent_impl(query, client, search_type)


@tool
def arxiv_research_agent(query: str) -> str:
    """
    Specialized agent for searching Arxiv database for scientific papers.

    Args:
        query: The search query for Arxiv

    Returns:
        Summarized findings from Arxiv papers
    """
    client = _session_manager.get_client("arxiv")
    if client is None:
        return "Error: Arxiv client session not available"
    return arxiv_research_agent_impl(query, client)


@tool
def pubmed_research_agent(query: str) -> str:
    """
    Specialized agent for searching PubMed database for medical papers.

    Args:
        query: The search query for PubMed

    Returns:
        Summarized findings from PubMed papers
    """
    client = _session_manager.get_client("pubmed")
    if client is None:
        return "Error: PubMed client session not available"
    return pubmed_research_agent_impl(query, client)


@tool
def chembl_research_agent(query: str) -> str:
    """
    Specialized agent for searching ChEMBL database for compound information.

    Args:
        query: The search query for ChEMBL

    Returns:
        Structured information about compounds, targets, and bioactivity
    """
    client = _session_manager.get_client("chembl")
    if client is None:
        return "Error: ChEMBL client session not available"
    return chembl_research_agent_impl(query, client)


@tool
def clinicaltrials_research_agent(query: str) -> str:
    """
    Specialized agent for searching ClinicalTrials.gov database.

    Args:
        query: The search query for ClinicalTrials.gov

    Returns:
        Information about relevant clinical trials
    """
    client = _session_manager.get_client("clinicaltrials")
    if client is None:
        return "Error: ClinicalTrials client session not available"
    return clinicaltrials_research_agent_impl(query, client)


#########################################################
# Orchestrator Agent - Multi-Agent Workflow
#########################################################
def create_orchestrator_agent(history_mode):
    # Orchestrator system prompt - updated to emphasize delegation to specialized agents
    system = """
    You are an orchestrator agent for drug discovery research. Your role is to coordinate a multi-agent workflow by delegating tasks to specialized research agents rather than using database tools directly.
    
    1. COORDINATION PHASE:
       - For simple queries: Answer directly WITHOUT using specialized tools
       - For complex research requests: Initiate the multi-agent research workflow using specialized agents
    
    2. PLANNING PHASE:
       - Use the planning_agent to determine which databases to search and with what queries
    
    3. EXECUTION PHASE:
       - ALWAYS delegate specialized search tasks to the appropriate research agents:
         - web_search_agent: For web searches and news (delegates to Tavily)
         - arxiv_research_agent: For scientific papers on ArXiv (delegates to ArXiv database)
         - pubmed_research_agent: For medical literature on PubMed (delegates to PubMed database)
         - chembl_research_agent: For compound data from ChEMBL (delegates to ChEMBL database)
         - clinicaltrials_research_agent: For clinical trial information (delegates to ClinicalTrials.gov)
       - DO NOT use database tools directly - always use the specialized agent tools
    
    4. SYNTHESIS PHASE:
       - Use the synthesis_agent to integrate findings into a comprehensive report
       - Generate a PDF report when appropriate using generate_pdf_report
    
    Your strength is in coordinating these specialized agents, not in directly accessing databases.
    Always provide a clear, professional response that addresses the researcher's needs.
    """

    model = get_model()

    try:
        # Use specialized agent tools instead of direct MCP tools
        tools = [
            planning_agent,
            web_search_agent,
            arxiv_research_agent,
            pubmed_research_agent,
            chembl_research_agent,
            clinicaltrials_research_agent,
            synthesis_agent,
            generate_pdf_report,
            file_write,
        ]

        if history_mode == "Enable":
            logger.info("history_mode: Enable")
            orchestrator = Agent(
                model=model,
                system_prompt=system,
                tools=tools,
                conversation_manager=conversation_manager,
            )
        else:
            logger.info("history_mode: Disable")
            orchestrator = Agent(model=model, system_prompt=system, tools=tools)

        return orchestrator
    except Exception as e:
        logger.error(f"Error initializing orchestrator agent: {e}")
        # If error occurs, substitute with a basic agent
        return Agent(
            model=model,
            system_prompt=system,
            tools=tools if "tools" in locals() else [],
        )


def run_multi_agent_system(question, history_mode, st):
    message_placeholder = st.empty()
    full_response = ""

    async def process_streaming_response():
        nonlocal full_response
        try:
            # Open all client sessions at once and manage them
            with tavily_mcp_client as tavily_client, arxiv_mcp_client as arxiv_client, pubmed_mcp_client as pubmed_client, chembl_mcp_client as chembl_client, clinicaltrials_mcp_client as clinicaltrials_client:

                # Create client session dictionary for distribution
                client_sessions = {
                    "tavily": tavily_client,
                    "arxiv": arxiv_client,
                    "pubmed": pubmed_client,
                    "chembl": chembl_client,
                    "clinicaltrials": clinicaltrials_client,
                }

                # Distribute active client sessions to specialized agent tools
                _session_manager.set_active_clients(client_sessions)

                # Log session distribution status
                session_status = _session_manager.get_session_status()
                logger.info(
                    f"MCP client session distribution status: {list(session_status.keys())}"
                )

                # Create orchestrator agent (clients are now accessible via session manager)
                current_orchestrator = create_orchestrator_agent(history_mode)

                # Stream the orchestrator response
                agent_stream = current_orchestrator.stream_async(question)
                async for event in agent_stream:
                    if "data" in event:
                        full_response += event["data"]
                        message_placeholder.markdown(full_response)

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            message_placeholder.markdown(
                "Sorry, an error occurred while generating the response."
            )
            logger.error(traceback.format_exc())  # Detailed error logging

    asyncio.run(process_streaming_response())

    return full_response
