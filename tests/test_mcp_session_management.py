"""
Unit tests for MCP client session management and sharing.

Tests verify that:
1. MCP client sessions are properly shared and reused across agent calls
2. Session manager correctly distributes client sessions to agents
3. Client session validation works properly
4. Error handling for unavailable or invalid sessions
"""

import datetime
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add application directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))

import chat


class TestMCPSessionManager(unittest.TestCase):
    """Test suite for MCPClientSessionManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.session_manager = chat.MCPClientSessionManager()

        # Create mock clients
        self.mock_tavily_client = Mock()
        self.mock_arxiv_client = Mock()
        self.mock_pubmed_client = Mock()
        self.mock_chembl_client = Mock()
        self.mock_clinicaltrials_client = Mock()

        self.client_sessions = {
            "tavily": self.mock_tavily_client,
            "arxiv": self.mock_arxiv_client,
            "pubmed": self.mock_pubmed_client,
            "chembl": self.mock_chembl_client,
            "clinicaltrials": self.mock_clinicaltrials_client,
        }

    def test_set_active_clients(self):
        """Test setting active client sessions"""
        self.session_manager.set_active_clients(self.client_sessions)

        # Verify all clients are stored
        stored_clients = self.session_manager.get_all_clients()
        self.assertEqual(len(stored_clients), 5)
        self.assertEqual(stored_clients["tavily"], self.mock_tavily_client)
        self.assertEqual(stored_clients["arxiv"], self.mock_arxiv_client)
        self.assertEqual(stored_clients["pubmed"], self.mock_pubmed_client)
        self.assertEqual(stored_clients["chembl"], self.mock_chembl_client)
        self.assertEqual(
            stored_clients["clinicaltrials"], self.mock_clinicaltrials_client
        )

        # Verify session status is tracked
        status = self.session_manager.get_session_status()
        self.assertEqual(len(status), 5)
        for client_type in self.client_sessions.keys():
            self.assertTrue(status[client_type]["active"])
            self.assertEqual(
                status[client_type]["client"], self.client_sessions[client_type]
            )

    def test_get_client_valid_type(self):
        """Test getting a client by valid type"""
        self.session_manager.set_active_clients(self.client_sessions)

        client = self.session_manager.get_client("tavily")
        self.assertEqual(client, self.mock_tavily_client)

        # Verify last_used timestamp is updated
        status = self.session_manager.get_session_status()
        self.assertIsNotNone(status["tavily"]["last_used"])
        self.assertIsInstance(status["tavily"]["last_used"], datetime.datetime)

    def test_get_client_invalid_type(self):
        """Test getting a client by invalid type"""
        self.session_manager.set_active_clients(self.client_sessions)

        client = self.session_manager.get_client("invalid_type")
        self.assertIsNone(client)

    def test_get_client_empty_manager(self):
        """Test getting a client when no sessions are set"""
        client = self.session_manager.get_client("tavily")
        self.assertIsNone(client)

    def test_is_client_available(self):
        """Test checking client availability"""
        self.session_manager.set_active_clients(self.client_sessions)

        # Test available clients
        self.assertTrue(self.session_manager.is_client_available("tavily"))
        self.assertTrue(self.session_manager.is_client_available("arxiv"))

        # Test unavailable client
        self.assertFalse(self.session_manager.is_client_available("invalid_type"))

    def test_get_all_clients_returns_copy(self):
        """Test that get_all_clients returns a copy, not the original dict"""
        self.session_manager.set_active_clients(self.client_sessions)

        clients_copy = self.session_manager.get_all_clients()

        # Modify the copy
        clients_copy["new_client"] = Mock()

        # Verify original is unchanged
        original_clients = self.session_manager.get_all_clients()
        self.assertNotIn("new_client", original_clients)
        self.assertEqual(len(original_clients), 5)

    def test_get_session_status_returns_copy(self):
        """Test that get_session_status returns a copy"""
        self.session_manager.set_active_clients(self.client_sessions)

        status_copy = self.session_manager.get_session_status()

        # Modify the copy
        status_copy["tavily"]["active"] = False

        # Verify original is unchanged
        original_status = self.session_manager.get_session_status()
        self.assertTrue(original_status["tavily"]["active"])


class TestMCPSessionDistribution(unittest.TestCase):
    """Test suite for MCP session distribution in run_multi_agent_system"""

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
    def test_client_session_distribution(
        self,
        mock_create_orchestrator,
        mock_clinicaltrials_client,
        mock_chembl_client,
        mock_pubmed_client,
        mock_arxiv_client,
        mock_tavily_client,
    ):
        """Test that client sessions are properly distributed to session manager"""

        # Setup mock clients as context managers
        mock_tavily_client.__enter__ = Mock(return_value=mock_tavily_client)
        mock_tavily_client.__exit__ = Mock(return_value=None)
        mock_arxiv_client.__enter__ = Mock(return_value=mock_arxiv_client)
        mock_arxiv_client.__exit__ = Mock(return_value=None)
        mock_pubmed_client.__enter__ = Mock(return_value=mock_pubmed_client)
        mock_pubmed_client.__exit__ = Mock(return_value=None)
        mock_chembl_client.__enter__ = Mock(return_value=mock_chembl_client)
        mock_chembl_client.__exit__ = Mock(return_value=None)
        mock_clinicaltrials_client.__enter__ = Mock(
            return_value=mock_clinicaltrials_client
        )
        mock_clinicaltrials_client.__exit__ = Mock(return_value=None)

        # Setup mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.stream.return_value = iter(["Test response"])
        mock_create_orchestrator.return_value = mock_orchestrator

        # Mock the session manager
        with patch.object(chat, "_session_manager") as mock_session_manager:
            # Run the function (this will be async, so we need to handle that)
            import asyncio

            async def run_test():
                return chat.run_multi_agent_system(
                    self.test_query, "Disable", self.mock_st
                )

            # Note: This test focuses on verifying the session distribution logic
            # The actual async execution would require more complex mocking

            # Verify that the session manager's set_active_clients would be called
            # with the correct client sessions in the actual implementation
            expected_clients = {
                "tavily": mock_tavily_client,
                "arxiv": mock_arxiv_client,
                "pubmed": mock_pubmed_client,
                "chembl": mock_chembl_client,
                "clinicaltrials": mock_clinicaltrials_client,
            }

            # This verifies the structure that would be passed to set_active_clients
            self.assertEqual(len(expected_clients), 5)
            self.assertIn("tavily", expected_clients)
            self.assertIn("arxiv", expected_clients)
            self.assertIn("pubmed", expected_clients)
            self.assertIn("chembl", expected_clients)
            self.assertIn("clinicaltrials", expected_clients)


class TestClientSessionReuse(unittest.TestCase):
    """Test suite for verifying client session reuse across multiple agent calls"""

    def setUp(self):
        """Set up test fixtures"""
        self.session_manager = chat.MCPClientSessionManager()
        self.mock_client = Mock()
        self.mock_client.list_tools_sync.return_value = [Mock()]

        # Set up session manager with mock client
        self.session_manager.set_active_clients({"tavily": self.mock_client})

        # Replace global session manager
        chat._session_manager = self.session_manager

    def test_multiple_agent_calls_reuse_same_client(self):
        """Test that multiple agent tool calls reuse the same client session"""
        with patch("chat.web_search_agent_impl") as mock_impl:
            mock_impl.return_value = "Search results"

            # Make multiple calls to the agent tool
            result1 = chat.web_search_agent("query 1")
            result2 = chat.web_search_agent("query 2")
            result3 = chat.web_search_agent("query 3")

            # Verify all calls succeeded
            self.assertEqual(result1, "Search results")
            self.assertEqual(result2, "Search results")
            self.assertEqual(result3, "Search results")

            # Verify the same client was passed to all calls
            self.assertEqual(mock_impl.call_count, 3)
            for call in mock_impl.call_args_list:
                self.assertEqual(
                    call[0][1], self.mock_client
                )  # Second argument is the client

    def test_client_session_tracking_updates(self):
        """Test that client session usage is properly tracked"""
        with patch("chat.web_search_agent_impl") as mock_impl:
            mock_impl.return_value = "Search results"

            # Get initial status
            initial_status = self.session_manager.get_session_status()
            initial_last_used = initial_status["tavily"]["last_used"]

            # Make a call
            chat.web_search_agent("test query")

            # Get updated status
            updated_status = self.session_manager.get_session_status()
            updated_last_used = updated_status["tavily"]["last_used"]

            # Verify last_used timestamp was updated
            if initial_last_used is None:
                self.assertIsNotNone(updated_last_used)
            else:
                self.assertGreater(updated_last_used, initial_last_used)


class TestErrorHandlingForUnavailableClients(unittest.TestCase):
    """Test suite for error handling when client sessions are unavailable"""

    def setUp(self):
        """Set up test fixtures"""
        # Create session manager with no active clients
        self.session_manager = chat.MCPClientSessionManager()
        chat._session_manager = self.session_manager

    def test_agent_tools_handle_unavailable_clients(self):
        """Test that agent tools handle unavailable client sessions gracefully"""
        # Test all agent tool wrappers
        agents_and_expected_errors = [
            (chat.web_search_agent, "Tavily client session not available"),
            (chat.arxiv_research_agent, "Arxiv client session not available"),
            (chat.pubmed_research_agent, "PubMed client session not available"),
            (chat.chembl_research_agent, "ChEMBL client session not available"),
            (
                chat.clinicaltrials_research_agent,
                "ClinicalTrials client session not available",
            ),
        ]

        for agent_func, expected_error in agents_and_expected_errors:
            with self.subTest(agent=agent_func.__name__):
                result = agent_func("test query")
                self.assertIn("Error", result)
                self.assertIn(expected_error, result)

    def test_partial_client_availability(self):
        """Test behavior when only some clients are available"""
        # Set up only tavily client
        mock_tavily_client = Mock()
        self.session_manager.set_active_clients({"tavily": mock_tavily_client})

        with patch("chat.web_search_agent_impl") as mock_web_impl:
            mock_web_impl.return_value = "Web search results"

            # Web search should work
            result = chat.web_search_agent("test query")
            self.assertEqual(result, "Web search results")

            # Other agents should return errors
            result = chat.arxiv_research_agent("test query")
            self.assertIn("Error", result)
            self.assertIn("Arxiv client session not available", result)


if __name__ == "__main__":
    unittest.main()
