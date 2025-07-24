#!/usr/bin/env python3

from engine import *
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCyclicMemoryFix(unittest.TestCase):
    """Tests for the cyclic memory reference fix that was causing infinite recursion."""

    def setUp(self):
        """Set up a fresh symbolic state for each test."""
        self.state = SymbolicState()

    def test_original_infinite_recursion_scenario(self):
        """Test the exact scenario that was causing infinite recursion."""
        engine = SymbolicEngine(self.state)
        trace = [
            "addq $1, (%rsp)",
            "mov (%rsp), %rax",
        ]

        engine.parse_trace_and_execute(trace)

        rax_var = self.state.current_var('rax')
        self.assertEqual(str(rax_var), "rax_1")

        expanded = expand_expr(rax_var, self.state)
        self.assertIsInstance(expanded, BinOp)
        self.assertEqual(expanded.op, "+")
        self.assertIsInstance(expanded.left, Mem)
        self.assertEqual(str(expanded.left), "mem[rsp_0]")
        self.assertIsInstance(expanded.right, Const)
        self.assertEqual(expanded.right.value, 1)

    def test_expand_cyclic_memory_direct(self):
        """Test direct cyclic memory reference expansion."""
        rsp_var = Var("rsp", 0)
        mem_rsp = Mem(rsp_var)
        cyclic_expr = BinOp("+", mem_rsp, Const(1))

        self.state.mem_store(rsp_var, cyclic_expr)

        expanded = expand_expr(mem_rsp, self.state)
        self.assertIsInstance(expanded, Mem)
        self.assertEqual(str(expanded), "mem[rsp_0]")

    def test_expand_non_cyclic_memory_still_works(self):
        """Test that non-cyclic memory expansion still works correctly."""
        addr = Const(0x1000)
        self.state.mem_store(addr, Const(42))

        mem_expr = Mem(addr)
        expanded = expand_expr(mem_expr, self.state)

        self.assertIsInstance(expanded, Const)
        self.assertEqual(expanded.value, 42)

    def test_optimization_after_expansion_fix(self):
        """Test that optimization works correctly after the expansion fix."""
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(["addq $1, (%rsp)", "mov (%rsp), %rax"])

        rax_var = self.state.current_var('rax')
        expanded = expand_expr(rax_var, self.state)

        optimized = optimize_expr(expanded, self.state)

        self.assertIsInstance(optimized, BinOp)

    def test_multiple_cycles_in_same_trace(self):
        """Test handling multiple potential cycles in the same trace."""
        engine = SymbolicEngine(self.state)
        trace = [
            "addq $1, (%rsp)",      # Creates potential cycle at mem[rsp_0]
            # Creates potential cycle at mem[(8 + rsp_0)]
            "addq $2, 8(%rsp)",
            "mov (%rsp), %rax",    # Load from first location
            "mov 8(%rsp), %rbx",   # Load from second location
        ]

        engine.parse_trace_and_execute(trace)

        rax_expanded = expand_expr(self.state.current_var('rax'), self.state)
        rbx_expanded = expand_expr(self.state.current_var('rbx'), self.state)

        self.assertIsNotNone(rax_expanded)
        self.assertIsNotNone(rbx_expanded)


if __name__ == '__main__':
    unittest.main()
