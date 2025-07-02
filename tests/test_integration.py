"""
Integration tests for the multi-agent orchestration system.

Tests verify:
1. End-to-end workflow from query to final report
2. Backward compatibility of run_multi_agent_system function
3. Streaming response functionality
4. Error handling scenarios in complete workflows
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add application directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))

import chat


class TestEndToEndWorkflow(unittest.TestCase):
    """Test suite for end-to-end multi-agent workflow"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "Research HER2 protein for drug discovery"
        self.mock_st = Mock()
        self.mock_st.empty.return_value = Mock()

    @patch("chat.tavily_mcp_client")
    @patch("chat.arxiv_mcp_client")
    @patch("chat.pubmed_mcp_client")
    @patch("chat.chembl_mcp_client")
    @patch("chat.clinicaltrials_mcp_client")
    @patch("chat.create_orchestrator_agent")
    def test_complete_workflow_execution(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test complete workflow from query to response"""

        # Setup mock clients as context managers
        for client in [
            mock_tavily_client,
            mock_arxiv_client,
            mock_pubmed_client,
            mock_chembl_client,
            mock_clinicaltrials_client,
        ]:
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

        # Setup mock orchestrator with streaming response
        mock_orchestrator = Mock()
        mock_orchestrator.stream.return_value = iter(
            [
                "Starting research on HER2 protein...",
                "Searching scientific databases...",
                "Analyzing results...",
                "Final comprehensive report on HER2 protein research.",
            ]
        )
        mock_create_orchestrator.return_value = mock_orchestrator

        # Mock the session manager
        with patch.object(chat, "_session_manager") as mock_session_manager:
            # Execute the workflow
            result = chat.run_multi_agent_system(
                self.test_query, "Disable", self.mock_st
            )

            # Verify orchestrator was created
            mock_create_orchestrator.assert_called_once_with("Disable")

            # Verify orchestrator was called with the query
            mock_orchestrator.stream.assert_called_once_with(self.test_query)

            # Verify session manager was configured with client sessions
            mock_session_manager.set_active_clients.assert_called_once()

            # Verify result contains expected content
            self.assertIn("HER2 protein", result)

    def test_run_multi_agent_system_signature_compatibility(self):
        """Test that run_multi_agent_system maintains backward compatible signature"""
        # Verify function exists and has expected signature
        self.assertTrue(hasattr(chat, "run_multi_agent_system"))

        func = chat.run_multi_agent_system

        # Check that function can be called with expected parameters
        # (We won't actually call it to avoid side effects, just verify signature)
        import inspect

        sig = inspect.signature(func)

        expected_params = ["question", "history_mode", "st"]
        actual_params = list(sig.parameters.keys())

        self.assertEqual(actual_params, expected_params)

    @patch("chat.tavily_mcp_client")
    @patch("chat.arxiv_mcp_client")
    @patch("chat.pubmed_mcp_client")
    @patch("chat.chembl_mcp_client")
    @patch("chat.clinicaltrials_mcp_client")
    @patch("chat.create_orchestrator_agent")
    def test_streaming_response_functionality(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test that streaming response functionality works correctly"""

        # Setup mock clients as context managers
        for client in [
            mock_tavily_client,
            mock_arxiv_client,
            mock_pubmed_client,
            mock_chembl_client,
            mock_clinicaltrials_client,
        ]:
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

        # Setup mock orchestrator with streaming chunks
        mock_orchestrator = Mock()
        stream_chunks = [
            "Chunk 1: Starting analysis",
            "Chunk 2: Searching databases",
            "Chunk 3: Synthesizing results",
            "Chunk 4: Final report",
        ]
        mock_orchestrator.stream.return_value = iter(stream_chunks)
        mock_create_orchestrator.return_value = mock_orchestrator

        # Setup mock streamlit components
        mock_message_placeholder = Mock()
        self.mock_st.empty.return_value = mock_message_placeholder

        with patch.object(chat, "_session_manager"):
            result = chat.run_multi_agent_system(
                self.test_query, "Enable", self.mock_st
            )

            # Verify streaming was used
            mock_orchestrator.stream.assert_called_once()

            # Verify result contains all chunks
            for chunk in stream_chunks:
                self.assertIn(chunk.split(": ")[1], result)  # Remove "Chunk X: " prefix


class TestErrorHandlingScenarios(unittest.TestCase):
    """Test suite for error handling in complete workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "Research HER2 protein"
        self.mock_st = Mock()
        self.mock_st.empty.return_value = Mock()

    @patch("chat.tavily_mcp_client")
    @patch("chat.arxiv_mcp_client")
    @patch("chat.pubmed_mcp_client")
    @patch("chat.chembl_mcp_client")
    @patch("chat.clinicaltrials_mcp_client")
    @patch("chat.create_orchestrator_agent")
    def test_orchestrator_creation_failure(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test handling of orchestrator creation failure"""

        # Setup mock clients as context managers
        for client in [
            mock_tavily_client,
            mock_arxiv_client,
            mock_pubmed_client,
            mock_chembl_client,
            mock_clinicaltrials_client,
        ]:
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

        # Make orchestrator creation fail
        mock_create_orchestrator.side_effect = Exception("Orchestrator creation failed")

        with patch.object(chat, "_session_manager"):
            # The function should handle the error gracefully
            try:
                result = chat.run_multi_agent_system(
                    self.test_query, "Disable", self.mock_st
                )
                # If no exception is raised, verify error is handled in result
                self.assertIn("error", result.lower())
            except Exception as e:
                # If exception is raised, it should be informative
                self.assertIn("Orchestrator", str(e))

    @patch("chat.tavily_mcp_client")
    @patch("chat.arxiv_mcp_client")
    @patch("chat.pubmed_mcp_client")
    @patch("chat.chembl_mcp_client")
    @patch("chat.clinicaltrials_mcp_client")
    @patch("chat.create_orchestrator_agent")
    def test_mcp_client_connection_failure(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test handling of MCP client connection failures"""

        # Make one client fail to connect
        mock_tavily_client.__enter__.side_effect = Exception("Connection failed")

        # Setup other clients normally
        for client in [
            mock_arxiv_client,
            mock_pubmed_client,
            mock_chembl_client,
            mock_clinicaltrials_client,
        ]:
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

        mock_orchestrator = Mock()
        mock_orchestrator.stream.return_value = iter(["Error handled gracefully"])
        mock_create_orchestrator.return_value = mock_orchestrator

        with patch.object(chat, "_session_manager"):
            # The function should handle client connection failures
            try:
                result = chat.run_multi_agent_system(
                    self.test_query, "Disable", self.mock_st
                )
                # Verify some response is still generated
                self.assertIsInstance(result, str)
            except Exception as e:
                # Connection failures should be handled gracefully
                self.assertIn("Connection", str(e))

    @patch("chat.tavily_mcp_client")
    @patch("chat.arxiv_mcp_client")
    @patch("chat.pubmed_mcp_client")
    @patch("chat.chembl_mcp_client")
    @patch("chat.clinicaltrials_mcp_client")
    @patch("chat.create_orchestrator_agent")
    def test_orchestrator_streaming_failure(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test handling of orchestrator streaming failures"""

        # Setup mock clients as context managers
        for client in [
            mock_tavily_client,
            mock_arxiv_client,
            mock_pubmed_client,
            mock_chembl_client,
            mock_clinicaltrials_client,
        ]:
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=None)

        # Make orchestrator streaming fail
        mock_orchestrator = Mock()
        mock_orchestrator.stream.side_effect = Exception("Streaming failed")
        mock_create_orchestrator.return_value = mock_orchestrator

        with patch.object(chat, "_session_manager"):
            # The function should handle streaming failures
            try:
                result = chat.run_multi_agent_system(
                    self.test_query, "Disable", self.mock_st
                )
                # Verify error is handled appropriately
                self.assertIn("error", result.lower())
            except Exception as e:
                # Streaming failures should be caught and handled
                self.assertIn("Streaming", str(e))


class TestBackwardCompatibility(unittest.TestCase):
    """Test suite for backward compatibility verification"""

    def test_function_interface_unchanged(self):
        """Test that the external interface of run_multi_agent_system is unchanged"""
        # Verify function signature
        import inspect

        sig = inspect.signature(chat.run_multi_agent_system)

        # Expected parameters based on original interface
        expected_params = ["question", "history_mode", "st"]
        actual_params = list(sig.parameters.keys())

        self.assertEqual(
            actual_params,
            expected_params,
            "run_multi_agent_system signature has changed",
        )

    def test_global_variables_exist(self):
        """Test that expected global variables still exist"""
        expected_globals = [
            "tavily_mcp_client",
            "arxiv_mcp_client",
            "pubmed_mcp_client",
            "chembl_mcp_client",
            "clinicaltrials_mcp_client",
            "conversation_manager",
        ]

        for var_name in expected_globals:
            self.assertTrue(
                hasattr(chat, var_name),
                f"Expected global variable '{var_name}' not found",
            )

    def test_agent_tool_functions_exist(self):
        """Test that all expected agent tool functions exist"""
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

        for tool_name in expected_tools:
            self.assertTrue(
                hasattr(chat, tool_name),
                f"Expected tool function '{tool_name}' not found",
            )

            # Verify it's decorated as a tool
            tool_func = getattr(chat, tool_name)
            self.assertTrue(
                hasattr(tool_func, "__wrapped__")
                or hasattr(tool_func, "_strands_tool"),
                f"Function '{tool_name}' is not properly decorated as a tool",
            )


if __name__ == "__main__":
    unittest.main()
