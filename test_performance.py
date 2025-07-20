#!/usr/bin/env python3

import time
import unittest
from engine import (
    SymbolicEngine, SymbolicState,
    expand_expr, optimize_expr, Var, Const, BinOp
)


class TestPerformance(unittest.TestCase):
    """Performance and scalability tests."""

    def test_large_trace_performance(self):
        """Test performance with a large trace."""
        print("\nTesting performance with large trace...")

        trace = []
        for i in range(1000):
            trace.append(f"mov ${i}, %rax")
            trace.append(f"add ${i+1}, %rax")
            trace.append("mov %rax, (%rsp)")

        engine = SymbolicEngine()

        start_time = time.time()
        engine.parse_trace_and_execute(trace)
        end_time = time.time()

        execution_time = end_time - start_time
        print(
            f"Executed {len(trace)} instructions in {execution_time:.3f} seconds")
        print(f"Rate: {len(trace)/execution_time:.1f} instructions/second")

        self.assertLess(execution_time, 5.0, "Execution took too long")

    def test_deep_expression_nesting(self):
        """Test performance with deeply nested expressions."""
        print("\nTesting deeply nested expressions...")

        engine = SymbolicEngine()

        # Create a chain of additions that will create deeply nested expressions
        trace = ["mov $1, %rax"]
        for i in range(100):
            trace.append(f"add $1, %rax")

        start_time = time.time()
        engine.parse_trace_and_execute(trace)
        end_time = time.time()

        print(
            f"Created nested expression in {end_time - start_time:.3f} seconds")

        # Test expansion performance
        final_var = engine.state.current_var("rax")

        start_time = time.time()
        expanded = expand_expr(final_var, engine.state)
        end_time = time.time()

        print(f"Expanded expression in {end_time - start_time:.3f} seconds")

        # Test optimization performance
        start_time = time.time()
        optimized = optimize_expr(final_var, engine.state)
        end_time = time.time()

        print(f"Optimized expression in {end_time - start_time:.3f} seconds")

    def test_memory_usage_pattern(self):
        """Test memory usage patterns with many memory operations."""
        print("\nTesting memory usage patterns...")

        engine = SymbolicEngine()

        trace = []
        for i in range(200):
            addr = i * 8  # Different addresses
            trace.append(f"mov ${i}, {addr}(%rsp)")
            trace.append(f"add $1, {addr}(%rsp)")

        start_time = time.time()
        engine.parse_trace_and_execute(trace)
        end_time = time.time()

        print(
            f"Processed {len(trace)} memory operations in {end_time - start_time:.3f} seconds")
        print(f"Memory locations used: {len(engine.state.mem)}")
        print(f"Definitions created: {len(engine.state.definitions)}")

    def test_register_version_explosion(self):
        """Test handling of many register versions."""
        print("\nTesting register version management...")

        engine = SymbolicEngine()

        trace = []
        for i in range(500):
            trace.append(f"mov ${i}, %rax")

        start_time = time.time()
        engine.parse_trace_and_execute(trace)
        end_time = time.time()

        print(
            f"Created {engine.state.reg_versions['rax']} register versions in {end_time - start_time:.3f} seconds")
        print(f"Total definitions: {len(engine.state.definitions)}")

    def test_optimization_effectiveness(self):
        """Test optimization effectiveness on various expression patterns."""
        print("\nTesting optimization effectiveness...")

        engine = SymbolicEngine()

        test_cases = [
            # Simple arithmetic that should fold
            BinOp("+", Const(10), Const(20)),
            BinOp("*", Const(5), Const(8)),
            BinOp("-", Const(100), Const(25)),

            # Identity operations
            BinOp("+", Var("rax", 1), Const(0)),
            BinOp("*", Var("rbx", 1), Const(1)),
            BinOp("-", Var("rcx", 1), Const(0)),

            # Zero multiplication
            BinOp("*", Var("rdx", 1), Const(0)),

            # Complex nested expression
            BinOp("+",
                  BinOp("*", Const(2), Const(5)),
                  BinOp("+", Const(0), Const(15)))
        ]

        for i, expr in enumerate(test_cases):
            start_time = time.time()
            optimized = optimize_expr(expr, engine.state)
            end_time = time.time()

            print(
                f"Case {i+1}: {expr} -> {optimized} ({end_time - start_time:.6f}s)")


class TestStressScenarios(unittest.TestCase):
    """Stress test scenarios and edge cases."""

    def test_alternating_register_usage(self):
        """Test alternating between many different registers."""
        engine = SymbolicEngine()

        registers = ["rax", "rbx", "rcx", "rdx",
                     "rsi", "rdi", "r8", "r9", "r10", "r11"]
        trace = []

        # Alternate between registers
        for i in range(100):
            reg = registers[i % len(registers)]
            trace.append(f"mov ${i}, %{reg}")
            trace.append(f"add $1, %{reg}")

        engine.parse_trace_and_execute(trace)

        # Check that all registers were used
        for reg in registers:
            self.assertIn(reg, engine.state.reg_versions)
            self.assertGreater(engine.state.reg_versions[reg], 0)

    def test_complex_addressing_modes(self):
        """Test various complex addressing modes."""
        engine = SymbolicEngine()

        # Set up base registers
        setup_trace = [
            "mov $0x1000, %rsp",
            "mov $0x2000, %rbp",
            "mov $4, %rax",
            "mov $8, %rbx"
        ]
        engine.parse_trace_and_execute(setup_trace)

        # Test complex addressing
        complex_trace = [
            "mov $100, (%rsp)",
            "mov $200, 8(%rsp)",
            "mov $300, -16(%rbp)",
            "mov $400, (%rsp,%rax,2)",
            "mov $500, 32(%rbp,%rbx,4)",
            "mov $600, -8(%rbp,%rax,1)"
        ]

        engine.parse_trace_and_execute(complex_trace)

        self.assertGreaterEqual(len(engine.state.mem), 5)

    def test_instruction_format_variations(self):
        """Test various instruction format variations."""
        engine = SymbolicEngine()

        # Test different instruction suffixes and formats
        trace = [
            "mov $42, %rax",      # No suffix
            "movl $43, %ebx",     # Long suffix with %e register
            "movq $44, %rcx",     # Quad suffix
            "addl $1, %eax",      # Add with %e register
            "addq $2, %rbx",      # Add with quad suffix
            "movl $100, (%rsp)",  # Memory with suffix
            "addl $10, 8(%rsp)"   # Memory add with suffix
        ]

        engine.parse_trace_and_execute(trace)

        self.assertIn("rax", engine.state.reg_versions)
        self.assertIn("rbx", engine.state.reg_versions)
        self.assertIn("rcx", engine.state.reg_versions)

    def test_edge_case_values(self):
        """Test edge case numeric values."""
        engine = SymbolicEngine()

        trace = [
            "mov $0, %rax",           # Zero
            "mov $-1, %rbx",          # Negative (if supported)
            "mov $0xffffffff, %rcx",  # Large hex
            "mov $2147483647, %rdx",  # Large decimal
            "add $0, %rax",           # Add zero
            "add $1, %rax",           # Add one
        ]

        engine.parse_trace_and_execute(trace)

        self.assertIn("rax", engine.state.reg_versions)

    def test_memory_address_collisions(self):
        """Test memory operations with address collisions."""
        engine = SymbolicEngine()

        trace = [
            "mov $0x1000, %rsp",
            "mov $8, %rax",
            "mov $100, 8(%rsp)",
            "mov $200, (%rsp,%rax,1)",
            "add $50, 8(%rsp)",
        ]

        engine.parse_trace_and_execute(trace)

        unique_addresses = set(str(addr) for addr in engine.state.mem.keys())

        self.assertLess(len(unique_addresses), 3)


def run_benchmarks():
    """Run performance benchmarks and display results."""
    print("="*60)
    print("SYMBOLIC EXECUTION ENGINE PERFORMANCE BENCHMARKS")
    print("="*60)

    suite = unittest.TestSuite()
    suite.addTest(TestPerformance('test_large_trace_performance'))
    suite.addTest(TestPerformance('test_deep_expression_nesting'))
    suite.addTest(TestPerformance('test_memory_usage_pattern'))
    suite.addTest(TestPerformance('test_register_version_explosion'))
    suite.addTest(TestPerformance('test_optimization_effectiveness'))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    print("\n" + "="*60)
    print("STRESS TEST SCENARIOS")
    print("="*60)

    stress_suite = unittest.TestSuite()
    stress_suite.addTest(TestStressScenarios(
        'test_alternating_register_usage'))
    stress_suite.addTest(TestStressScenarios('test_complex_addressing_modes'))
    stress_suite.addTest(TestStressScenarios(
        'test_instruction_format_variations'))
    stress_suite.addTest(TestStressScenarios('test_edge_case_values'))
    stress_suite.addTest(TestStressScenarios('test_memory_address_collisions'))

    runner.run(stress_suite)


if __name__ == '__main__':
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == '--benchmark':
        run_benchmarks()
    else:
        unittest.main(verbosity=2)
