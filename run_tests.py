import sys
import unittest
import time
from io import StringIO

try:
    import coverage
except ImportError:
    coverage = None


def run_test_suite(test_module_name, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    import importlib
    test_module = importlib.import_module(test_module_name)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)

    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)

    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    print(stream.getvalue())

    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0

    print(f"\nSUMMARY: {description}")
    print(f"  Tests run: {total_tests}")
    print(f"  Failures: {failures}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Time: {end_time - start_time:.2f} seconds")

    if failures > 0:
        print("  FAILURES:")
        for test, traceback in result.failures:
            print(
                f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if errors > 0:
        print("  ERRORS:")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")

    success = failures == 0 and errors == 0
    print(f"  Status: {'PASSED' if success else 'FAILED'}")

    return {
        'success': success,
        'total': total_tests,
        'failures': failures,
        'errors': errors,
        'skipped': skipped,
        'time': end_time - start_time
    }


def test_engine_functionality():
    """Test basic engine functionality with sample traces."""
    print(f"\n{'='*60}")
    print("TESTING ENGINE WITH SAMPLE TRACES")
    print(f"{'='*60}")

    from engine import SymbolicState, SymbolicEngine, optimize_expr

    trace_files = [
        ('test/test_traces/test_trace1.log', 'Basic arithmetic operations'),
        ('test/test_traces/test_trace2.log', 'Complex addressing modes'),
        ('test/test_traces/test_trace3.log', 'Mixed instructions with control flow')
    ]

    for filename, description in trace_files:
        print(f"\nTesting {filename}: {description}")
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            trace = [line.strip() for line in lines if line.strip()
                     and not line.startswith('#')]
            trace = [line.split(':')[1].strip()
                     for line in trace if ':' in line]

            print(f"  Instructions to process: {len(trace)}")

            state = SymbolicState()
            start_time = time.time()
            engine = SymbolicEngine(state)
            engine.parse_trace_and_execute(trace)
            end_time = time.time()

            print(f"  Execution time: {end_time - start_time:.4f} seconds")
            print(f"  Register versions created: {len(state.reg_versions)}")
            print(f"  Memory locations used: {len(state.mem)}")
            print(f"  Total definitions: {len(state.definitions)}")

            if 'rax' in state.reg_versions:
                final_rax = state.current_var('rax')
                optimized = optimize_expr(final_rax, state)
                print(f"  Final rax optimized: {optimized}")

            print(f"  Status: SUCCESS")

        except FileNotFoundError:
            print(f"  Status: SKIPPED (file not found)")
        except Exception as e:
            print(f"  Status: ERROR - {e}")


def main():
    """Main test runner."""
    print("SYMBOLIC EXECUTION ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    start_time = time.time()

    test_suites = [
        ('test.test_engine', 'CORE ENGINE TESTS'),
        ('test.test_performance', 'PERFORMANCE AND STRESS TESTS'),
        ('test.test_cyclic_memory_fix', 'CYCLIC MEMORY FIX TESTS'),
        ('test.test_parser', 'TRACE PARSER TESTS'),
    ]

    all_results = []
    overall_success = True

    if coverage:
        cov = coverage.Coverage()
        cov.start()

    for module_name, description in test_suites:
        try:
            result = run_test_suite(module_name, description)
            all_results.append((description, result))
            if not result['success']:
                overall_success = False
        except ImportError as e:
            print(f"\nERROR: Could not import {module_name}: {e}")
            overall_success = False
        except Exception as e:
            print(f"\nERROR running {description}: {e}")
            overall_success = False

    test_engine_functionality()

    end_time = time.time()
    total_time = end_time - start_time

    if coverage:
        cov.stop()
        cov.save()
        cov.report()
        cov.xml_report(outfile='coverage.xml')
        print("\nCoverage report saved to coverage.xml")

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    for description, result in all_results:
        total_tests += result['total']
        total_failures += result['failures']
        total_errors += result['errors']
        total_skipped += result['skipped']

        status = "PASSED" if result['success'] else "FAILED"
        print(
            f"{description}: {status} ({result['total']} tests, {result['time']:.2f}s)")

    print(f"\nTOTAL STATISTICS:")
    print(f"  Tests run: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Overall status: {'PASSED' if overall_success else 'FAILED'}")

    return 0 if overall_success else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        result = run_test_suite('test.test_engine', 'QUICK ENGINE TESTS')
        sys.exit(0 if result['success'] else 1)
    else:
        sys.exit(main())
