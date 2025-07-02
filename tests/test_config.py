"""
Test configuration and utilities for multi-agent orchestration system tests.
"""

import os
import sys
import unittest
from unittest.mock import Mock

# Test configuration
TEST_CONFIG = {
    "mock_responses": {
        "web_search": "Web search results for query",
        "arxiv_search": "Arxiv research findings",
        "pubmed_search": "PubMed medical literature results",
        "chembl_search": "ChEMBL compound information",
        "clinicaltrials_search": "Clinical trials data",
    },
    "test_queries": {
        "simple": "What is HER2?",
        "complex": "Research HER2 protein for drug discovery including compounds and clinical trials",
        "invalid": "",
        "long": "A" * 1000,
    },
}


def create_mock_mcp_client():
    """Create a mock MCP client for testing"""
    mock_client = Mock()
    mock_client.list_tools_sync.return_value = [
        Mock(name="search_tool"),
        Mock(name="query_tool"),
    ]
    return mock_client


def create_mock_streamlit():
    """Create a mock Streamlit object for testing"""
    mock_st = Mock()
    mock_st.empty.return_value = Mock()
    return mock_st


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup for all tests"""

    def setUp(self):
        """Common setup for all tests"""
        self.test_config = TEST_CONFIG
        self.mock_client = create_mock_mcp_client()
        self.mock_st = create_mock_streamlit()

    def assertContainsError(self, result, error_type=None):
        """Assert that result contains an error message"""
        self.assertIsInstance(result, str)
        self.assertIn("Error", result)
        if error_type:
            self.assertIn(error_type, result)
