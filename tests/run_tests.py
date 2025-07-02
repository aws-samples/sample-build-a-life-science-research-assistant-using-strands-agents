#!/usr/bin/env python3
"""
Test runner for the multi-agent orchestration system tests.

This script runs all test suites and provides a comprehensive report
of the test results for the refactored multi-agent system.
"""

import logging
import os
import sys
import unittest

# Configure logging to reduce noise during testing
logging.basicConfig(level=logging.ERROR)

# Add application directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "application"))


def run_all_tests():
    """Run all test suites and return results"""

    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result


def print_test_summary(result):
    """Print a summary of test results"""
    print("\n" + "=" * 60)
    print("MULTI-AGENT ORCHESTRATION SYSTEM TEST SUMMARY")
    print("=" * 60)

    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            newline = "\n"
            print(
                f"- {test}: {traceback.split('AssertionError: ')[-1].split(newline)[0]}"
            )

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            newline = "\n"
            print(f"- {test}: {traceback.split(newline)[-2]}")

    success_rate = (
        (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            * 100
        )
        if result.testsRun > 0
        else 0
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        print("The refactored multi-agent system is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the failures and errors above.")

    print("=" * 60)


def main():
    """Main test runner function"""
    print("Running Multi-Agent Orchestration System Tests...")
    print("This will test:")
    print("1. Specialized agent functions")
    print("2. Orchestrator delegation behavior")
    print("3. MCP client session management")
    print("4. End-to-end integration")
    print("5. Error handling scenarios")
    print("6. Backward compatibility")
    print("\n" + "-" * 60)

    try:
        result = run_all_tests()
        print_test_summary(result)

        # Return appropriate exit code
        return 0 if result.wasSuccessful() else 1

    except Exception as e:
        print(f"\nFATAL ERROR: Failed to run tests: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
