"""
Unit tests for orchestrator agent delegation to specialized agents.

Tests verify that the orchestrator:
1. Properly delegates to specialized agents instead of using MCP tools directly
2. Uses the correct agent tools for different types of queries
3. Maintains proper workflow coordination
4. Handles agent failures gracefully
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add application directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))

import chat


class TestOrchestratorDelegation(unittest.TestCase):
    """Test suite for orchestrator agent delegation behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "Research HER2 protein for drug discovery"

    def test_create_orchestrator_agent_uses_specialized_tools(self):
        """Test that orchestrator is created with specialized agent tools, not MCP tools"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Create orchestrator
            orchestrator = chat.create_orchestrator_agent("Disable")

            # Verify Agent was called
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args

            # Verify system prompt emphasizes delegation
            system_prompt = call_args[1]["system_prompt"]
            self.assertIn("delegate", system_prompt.lower())
            self.assertIn("specialized", system_prompt.lower())
            self.assertIn("orchestrator", system_prompt.lower())

            # Verify tools include specialized agents
            tools = call_args[1]["tools"]
            tool_names = [
                tool.__name__ if hasattr(tool, "__name__") else str(tool)
                for tool in tools
            ]

            # Check that specialized agent tools are included
            expected_tools = [
                "planning_agent",
                "web_search_agent",
                "arxiv_research_agent",
                "pubmed_research_agent",
                "chembl_research_agent",
                "clinicaltrials_research_agent",
                "synthesis_agent",
                "generate_pdf_report",
            ]

            for expected_tool in expected_tools:
                self.assertTrue(
                    any(expected_tool in tool_name for tool_name in tool_names),
                    f"Expected tool '{expected_tool}' not found in orchestrator tools",
                )

    def test_orchestrator_system_prompt_emphasizes_delegation(self):
        """Test that orchestrator system prompt emphasizes delegation over direct MCP tool usage"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent_class.return_value = Mock()

            chat.create_orchestrator_agent("Disable")

            call_args = mock_agent_class.call_args
            system_prompt = call_args[1]["system_prompt"].lower()

            # Verify delegation emphasis
            self.assertIn("delegate", system_prompt)
            self.assertIn("specialized research agents", system_prompt)
            self.assertIn("do not use database tools directly", system_prompt)
            self.assertIn("always use the specialized agent tools", system_prompt)

            # Verify specific agent mentions
            self.assertIn("web_search_agent", system_prompt)
            self.assertIn("arxiv_research_agent", system_prompt)
            self.assertIn("pubmed_research_agent", system_prompt)
            self.assertIn("chembl_research_agent", system_prompt)
            self.assertIn("clinicaltrials_research_agent", system_prompt)

    def test_orchestrator_with_history_mode_enabled(self):
        """Test orchestrator creation with conversation history enabled"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent_class.return_value = Mock()

            chat.create_orchestrator_agent("Enable")

            call_args = mock_agent_class.call_args

            # Verify conversation manager is included when history is enabled
            self.assertIn("conversation_manager", call_args[1])
            self.assertEqual(
                call_args[1]["conversation_manager"], chat.conversation_manager
            )

    def test_orchestrator_with_history_mode_disabled(self):
        """Test orchestrator creation with conversation history disabled"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent_class.return_value = Mock()

            chat.create_orchestrator_agent("Disable")

            call_args = mock_agent_class.call_args

            # Verify conversation manager is not included when history is disabled
            self.assertNotIn("conversation_manager", call_args[1])

    def test_orchestrator_error_handling_during_creation(self):
        """Test orchestrator handles errors during creation gracefully"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model

            # First call fails, second call succeeds (fallback)
            mock_agent_class.side_effect = [Exception("Agent creation failed"), Mock()]

            result = chat.create_orchestrator_agent("Disable")

            # Verify fallback agent was created
            self.assertIsNotNone(result)
            self.assertEqual(mock_agent_class.call_count, 2)

    @patch("chat.logger")
    def test_orchestrator_logs_history_mode(self, mock_logger):
        """Test that orchestrator logs the history mode setting"""
        with patch("chat.get_model") as mock_get_model, patch(
            "chat.Agent"
        ) as mock_agent_class:

            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent_class.return_value = Mock()

            # Test with history enabled
            chat.create_orchestrator_agent("Enable")
            mock_logger.info.assert_any_call("history_mode: Enable")

            # Test with history disabled
            chat.create_orchestrator_agent("Disable")
            mock_logger.info.assert_any_call("history_mode: Disable")


class TestAgentToolWrappers(unittest.TestCase):
    """Test suite for agent tool wrapper functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "test query"

        # Mock the session manager
        self.mock_session_manager = Mock()
        chat._session_manager = self.mock_session_manager

    def test_web_search_agent_tool_wrapper(self):
        """Test web_search_agent tool wrapper delegates to implementation"""
        mock_client = Mock()
        self.mock_session_manager.get_client.return_value = mock_client

        with patch("chat.web_search_agent_impl") as mock_impl:
            mock_impl.return_value = "Web search results"

            result = chat.web_search_agent(self.test_query, "news")

            # Verify session manager was called for correct client type
            self.mock_session_manager.get_client.assert_called_once_with("tavily")

            # Verify implementation was called with correct parameters
            mock_impl.assert_called_once_with(self.test_query, mock_client, "news")

            # Verify result
            self.assertEqual(result, "Web search results")

    def test_web_search_agent_tool_wrapper_no_client(self):
        """Test web_search_agent tool wrapper when client is unavailable"""
        self.mock_session_manager.get_client.return_value = None

        result = chat.web_search_agent(self.test_query)

        self.assertEqual(result, "Error: Tavily client session not available")

    def test_arxiv_research_agent_tool_wrapper(self):
        """Test arxiv_research_agent tool wrapper delegates to implementation"""
        mock_client = Mock()
        self.mock_session_manager.get_client.return_value = mock_client

        with patch("chat.arxiv_research_agent_impl") as mock_impl:
            mock_impl.return_value = "Arxiv results"

            result = chat.arxiv_research_agent(self.test_query)

            self.mock_session_manager.get_client.assert_called_once_with("arxiv")
            mock_impl.assert_called_once_with(self.test_query, mock_client)
            self.assertEqual(result, "Arxiv results")

    def test_pubmed_research_agent_tool_wrapper(self):
        """Test pubmed_research_agent tool wrapper delegates to implementation"""
        mock_client = Mock()
        self.mock_session_manager.get_client.return_value = mock_client

        with patch("chat.pubmed_research_agent_impl") as mock_impl:
            mock_impl.return_value = "PubMed results"

            result = chat.pubmed_research_agent(self.test_query)

            self.mock_session_manager.get_client.assert_called_once_with("pubmed")
            mock_impl.assert_called_once_with(self.test_query, mock_client)
            self.assertEqual(result, "PubMed results")

    def test_chembl_research_agent_tool_wrapper(self):
        """Test chembl_research_agent tool wrapper delegates to implementation"""
        mock_client = Mock()
        self.mock_session_manager.get_client.return_value = mock_client

        with patch("chat.chembl_research_agent_impl") as mock_impl:
            mock_impl.return_value = "ChEMBL results"

            result = chat.chembl_research_agent(self.test_query)

            self.mock_session_manager.get_client.assert_called_once_with("chembl")
            mock_impl.assert_called_once_with(self.test_query, mock_client)
            self.assertEqual(result, "ChEMBL results")

    def test_clinicaltrials_research_agent_tool_wrapper(self):
        """Test clinicaltrials_research_agent tool wrapper delegates to implementation"""
        mock_client = Mock()
        self.mock_session_manager.get_client.return_value = mock_client

        with patch("chat.clinicaltrials_research_agent_impl") as mock_impl:
            mock_impl.return_value = "ClinicalTrials results"

            result = chat.clinicaltrials_research_agent(self.test_query)

            self.mock_session_manager.get_client.assert_called_once_with(
                "clinicaltrials"
            )
            mock_impl.assert_called_once_with(self.test_query, mock_client)
            self.assertEqual(result, "ClinicalTrials results")


if __name__ == "__main__":
    unittest.main()
