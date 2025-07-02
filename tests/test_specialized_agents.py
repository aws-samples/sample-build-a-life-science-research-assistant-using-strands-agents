"""
Unit tests for specialized agent functions in the multi-agent orchestration system.

Tests verify that each specialized agent function:
1. Properly accepts and uses MCP client sessions
2. Returns structured results
3. Handles error scenarios appropriately
4. Validates client session requirements
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add application directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'application'))

import chat


class TestSpecializedAgents(unittest.TestCase):
    """Test suite for specialized agent functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.mock_tools = [Mock(name="mock_tool_1"), Mock(name="mock_tool_2")]
        self.mock_client.list_tools_sync.return_value = self.mock_tools
        
        self.test_query = "test query about drug discovery"
        
    def test_web_search_agent_impl_with_valid_client(self):
        """Test web_search_agent_impl with valid client session"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            # Setup mocks
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "Web search results"
            mock_agent_class.return_value = mock_agent
            
            # Test the function
            result = chat.web_search_agent_impl(self.test_query, self.mock_client)
            
            # Verify client validation was called
            self.mock_client.list_tools_sync.assert_called_once()
            
            # Verify agent was created with correct parameters
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            self.assertEqual(call_args[1]['model'], mock_model)
            self.assertEqual(call_args[1]['tools'], self.mock_tools)
            self.assertIn('web search agent', call_args[1]['system_prompt'].lower())
            
            # Verify agent was called with query
            mock_agent.assert_called_once_with(self.test_query)
            
            # Verify result
            self.assertEqual(result, "Web search results")
    
    def test_web_search_agent_impl_with_none_client(self):
        """Test web_search_agent_impl with None client session"""
        result = chat.web_search_agent_impl(self.test_query, None)
        
        self.assertIn("Error", result)
        self.assertIn("Active Tavily client session is required", result)
    
    def test_web_search_agent_impl_with_invalid_client(self):
        """Test web_search_agent_impl with invalid client session"""
        invalid_client = Mock()
        invalid_client.list_tools_sync.return_value = []  # No tools available
        
        result = chat.web_search_agent_impl(self.test_query, invalid_client)
        
        self.assertIn("Error", result)
        self.assertIn("invalid or has no available tools", result)
    
    def test_web_search_agent_impl_with_search_types(self):
        """Test web_search_agent_impl with different search types"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "Search results"
            mock_agent_class.return_value = mock_agent
            
            # Test news search
            chat.web_search_agent_impl(self.test_query, self.mock_client, "news")
            mock_agent.assert_called_with(f"RECENT NEWS: {self.test_query}")
            
            # Test answer search
            chat.web_search_agent_impl(self.test_query, self.mock_client, "answer")
            mock_agent.assert_called_with(f"DIRECT ANSWER: {self.test_query}")
    
    def test_arxiv_research_agent_impl_with_valid_client(self):
        """Test arxiv_research_agent_impl with valid client session"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "Arxiv research results"
            mock_agent_class.return_value = mock_agent
            
            result = chat.arxiv_research_agent_impl(self.test_query, self.mock_client)
            
            # Verify client validation
            self.mock_client.list_tools_sync.assert_called_once()
            
            # Verify agent creation
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            self.assertIn('arxiv research agent', call_args[1]['system_prompt'].lower())
            
            # Verify result
            self.assertEqual(result, "Arxiv research results")
    
    def test_arxiv_research_agent_impl_with_none_client(self):
        """Test arxiv_research_agent_impl with None client session"""
        result = chat.arxiv_research_agent_impl(self.test_query, None)
        
        self.assertIn("Error", result)
        self.assertIn("Active Arxiv client session is required", result)
    
    def test_pubmed_research_agent_impl_with_valid_client(self):
        """Test pubmed_research_agent_impl with valid client session"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "PubMed research results"
            mock_agent_class.return_value = mock_agent
            
            result = chat.pubmed_research_agent_impl(self.test_query, self.mock_client)
            
            # Verify client validation
            self.mock_client.list_tools_sync.assert_called_once()
            
            # Verify agent creation
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            self.assertIn('pubmed research agent', call_args[1]['system_prompt'].lower())
            
            # Verify result
            self.assertEqual(result, "PubMed research results")
    
    def test_pubmed_research_agent_impl_with_none_client(self):
        """Test pubmed_research_agent_impl with None client session"""
        result = chat.pubmed_research_agent_impl(self.test_query, None)
        
        self.assertIn("Error", result)
        self.assertIn("Active PubMed client session is required", result)
    
    def test_chembl_research_agent_impl_with_valid_client(self):
        """Test chembl_research_agent_impl with valid client session"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "ChEMBL research results"
            mock_agent_class.return_value = mock_agent
            
            result = chat.chembl_research_agent_impl(self.test_query, self.mock_client)
            
            # Verify client validation
            self.mock_client.list_tools_sync.assert_called_once()
            
            # Verify agent creation
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            self.assertIn('chembl research agent', call_args[1]['system_prompt'].lower())
            
            # Verify result
            self.assertEqual(result, "ChEMBL research results")
    
    def test_chembl_research_agent_impl_with_none_client(self):
        """Test chembl_research_agent_impl with None client session"""
        result = chat.chembl_research_agent_impl(self.test_query, None)
        
        self.assertIn("Error", result)
        self.assertIn("Active ChEMBL client session is required", result)
    
    def test_clinicaltrials_research_agent_impl_with_valid_client(self):
        """Test clinicaltrials_research_agent_impl with valid client session"""
        with patch('chat.get_model') as mock_get_model, \
             patch('chat.Agent') as mock_agent_class:
            
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            mock_agent = Mock()
            mock_agent.return_value = "ClinicalTrials research results"
            mock_agent_class.return_value = mock_agent
            
            result = chat.clinicaltrials_research_agent_impl(self.test_query, self.mock_client)
            
            # Verify client validation
            self.mock_client.list_tools_sync.assert_called_once()
            
            # Verify agent creation
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            self.assertIn('clinicaltrials.gov research agent', call_args[1]['system_prompt'].lower())
            
            # Verify result
            self.assertEqual(result, "ClinicalTrials research results")
    
    def test_clinicaltrials_research_agent_impl_with_none_client(self):
        """Test clinicaltrials_research_agent_impl with None client session"""
        result = chat.clinicaltrials_research_agent_impl(self.test_query, None)
        
        self.assertIn("Error", result)
        self.assertIn("Active ClinicalTrials client session is required", result)
    
    def test_agent_exception_handling(self):
        """Test that agents handle exceptions properly"""
        with patch('chat.get_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Model initialization failed")
            
            result = chat.web_search_agent_impl(self.test_query, self.mock_client)
            
            self.assertIn("Error in web search agent", result)
            self.assertIn("Model initialization failed", result)


if __name__ == '__main__':
    unittest.main()