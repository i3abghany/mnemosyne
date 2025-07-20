#!/usr/bin/env python3

import unittest
import tempfile
import os
from engine import (
    Expr, Const, Var, Mem, BinOp, SymbolicState,
    parse_address, SymbolicEngine,
    expand_expr, optimize_expr
)


class TestExpressions(unittest.TestCase):
    """Test the expression hierarchy and string representations."""

    def test_const_creation_and_str(self):
        """Test Const expression creation and string representation."""
        c1 = Const(42)
        self.assertEqual(c1.value, 42)
        self.assertEqual(str(c1), "42")

        c2 = Const(-10)
        self.assertEqual(str(c2), "-10")

        c3 = Const(0)
        self.assertEqual(str(c3), "0")

    def test_var_creation_and_str(self):
        """Test Var expression creation and string representation."""
        v1 = Var("rax", 1)
        self.assertEqual(v1.name, "rax")
        self.assertEqual(v1.version, 1)
        self.assertEqual(str(v1), "rax_1")

        v2 = Var("rbx", 10)
        self.assertEqual(str(v2), "rbx_10")

    def test_mem_creation_and_str(self):
        """Test Mem expression creation and string representation."""
        addr = Const(4096)
        m1 = Mem(addr)
        self.assertEqual(str(m1), "mem[4096]")

        addr2 = BinOp("+", Var("rsp", 1), Const(8))
        m2 = Mem(addr2)
        self.assertEqual(str(m2), "mem[(rsp_1 + 8)]")

    def test_binop_creation_and_str(self):
        """Test BinOp expression creation and string representation."""
        left = Var("rax", 1)
        right = Const(42)

        add = BinOp("+", left, right)
        self.assertEqual(str(add), "(rax_1 + 42)")

        sub = BinOp("-", left, right)
        self.assertEqual(str(sub), "(rax_1 - 42)")

        mul = BinOp("*", left, right)
        self.assertEqual(str(mul), "(rax_1 * 42)")

    def test_nested_binop(self):
        """Test nested binary operations."""
        # (rax_1 + (rbx_2 * 4))
        inner = BinOp("*", Var("rbx", 2), Const(4))
        outer = BinOp("+", Var("rax", 1), inner)
        self.assertEqual(str(outer), "(rax_1 + (rbx_2 * 4))")


class TestSymbolicState(unittest.TestCase):
    """Test the SymbolicState class functionality."""

    def setUp(self):
        """Set up a fresh symbolic state for each test."""
        self.state = SymbolicState()

    def test_initial_state(self):
        """Test initial state setup."""
        self.assertEqual(self.state.reg_versions, {})
        self.assertEqual(len(self.state.mem), 0)
        self.assertIn("rsp_1", self.state.definitions)
        self.assertEqual(str(self.state.rsp_val), "4096")  # 0x1000

    def test_new_var_creation(self):
        """Test creating new variables."""
        v1 = self.state.new_var("rax")
        self.assertEqual(str(v1), "rax_1")
        self.assertEqual(self.state.reg_versions["rax"], 1)

        v2 = self.state.new_var("rax")
        self.assertEqual(str(v2), "rax_2")
        self.assertEqual(self.state.reg_versions["rax"], 2)

        v3 = self.state.new_var("rbx")
        self.assertEqual(str(v3), "rbx_1")
        self.assertEqual(self.state.reg_versions["rbx"], 1)

    def test_current_var(self):
        """Test getting current variable versions."""
        # Before any writes
        v1 = self.state.current_var("rax")
        self.assertEqual(str(v1), "rax_0")

        # After creating a new version
        self.state.new_var("rax")
        v2 = self.state.current_var("rax")
        self.assertEqual(str(v2), "rax_1")

    def test_write_reg(self):
        """Test register writes."""
        expr = Const(42)
        var, returned_expr = self.state.write_reg("rax", expr)

        self.assertEqual(str(var), "rax_1")
        self.assertEqual(returned_expr, expr)
        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(self.state.definitions["rax_1"], expr)

    def test_read_reg(self):
        """Test register reads."""
        # Write then read
        self.state.write_reg("rax", Const(42))
        var = self.state.read_reg("rax")
        self.assertEqual(str(var), "rax_1")

    def test_mem_store_and_load(self):
        """Test memory operations."""
        addr = BinOp("+", Var("rsp", 0), Const(8))
        value = Const(100)

        # Store
        mem_var, returned_value = self.state.mem_store(addr, value)
        self.assertEqual(returned_value, value)
        self.assertIn(str(addr), self.state.mem)
        self.assertEqual(self.state.mem[str(addr)], value)

        # Load existing
        loaded = self.state.mem_load(addr)
        self.assertEqual(loaded, value)

        # Load non-existing
        new_addr = BinOp("+", Var("rsp", 0), Const(16))
        loaded_new = self.state.mem_load(new_addr)
        self.assertIsInstance(loaded_new, Mem)
        self.assertEqual(str(loaded_new), f"mem[{str(new_addr)}]")


class TestAddressParsing(unittest.TestCase):
    """Test address parsing functionality."""

    def setUp(self):
        """Set up a symbolic state for address parsing tests."""
        self.state = SymbolicState()
        # Set up some register values
        self.state.write_reg("rsp", Const(0x1000))
        self.state.write_reg("rbp", Const(0x2000))
        self.state.write_reg("rax", Const(0x3000))

    def test_parse_simple_register_indirect(self):
        """Test parsing simple register indirect addressing."""
        # (%rsp)
        addr = parse_address("(%rsp)", self.state)
        self.assertIsInstance(addr, BinOp)
        self.assertEqual(addr.op, "+")
        self.assertEqual(str(addr.left), "rsp_1")
        self.assertEqual(str(addr.right), "0")

    def test_parse_displacement_only(self):
        """Test parsing displacement-only addressing."""
        # 8(%rsp)
        addr = parse_address("8(%rsp)", self.state)
        self.assertIsInstance(addr, BinOp)
        self.assertEqual(addr.op, "+")
        self.assertEqual(str(addr.left), "rsp_1")
        self.assertEqual(str(addr.right), "8")

        # -16(%rbp)
        addr2 = parse_address("-16(%rbp)", self.state)
        self.assertEqual(str(addr2.right), "-16")

    def test_parse_hex_displacement(self):
        """Test parsing hexadecimal displacements."""
        # 0x10(%rsp)
        addr = parse_address("0x10(%rsp)", self.state)
        self.assertEqual(str(addr.right), "16")

        # -0x8(%rbp)
        addr2 = parse_address("-0x8(%rbp)", self.state)
        self.assertEqual(str(addr2.right), "-8")

    def test_parse_index_with_scale(self):
        """Test parsing indexed addressing with scale."""
        # (%rsp,%rax,4)
        addr = parse_address("(%rsp,%rax,4)", self.state)
        self.assertIsInstance(addr, BinOp)
        # Should have rsp + (rax * 4)
        self.assertEqual(str(addr.left), "rsp_1")
        self.assertIsInstance(addr.right, BinOp)
        self.assertEqual(addr.right.op, "*")
        self.assertEqual(str(addr.right.left), "rax_1")
        self.assertEqual(str(addr.right.right), "4")

    def test_parse_complex_addressing(self):
        """Test parsing complex addressing modes."""
        # 8(%rsp,%rax,2)
        addr = parse_address("8(%rsp,%rax,2)", self.state)
        # Should be ((rsp + (rax * 2)) + 8) - order may vary due to implementation
        addr_str = str(addr)
        self.assertIn("rsp_1", addr_str)
        self.assertIn("rax_1", addr_str)
        self.assertIn("2", addr_str)
        self.assertIn("8", addr_str)

    def test_parse_index_without_scale(self):
        """Test parsing indexed addressing without explicit scale."""
        # (%rsp,%rax)
        addr = parse_address("(%rsp,%rax)", self.state)
        addr_str = str(addr)
        self.assertIn("rsp_1", addr_str)
        self.assertIn("rax_1", addr_str)

    def test_parse_invalid_addresses(self):
        """Test error handling for invalid address formats."""
        with self.assertRaises(ValueError):
            parse_address("invalid", self.state)

        with self.assertRaises(ValueError):
            parse_address("()", self.state)


class TestTraceExecution(unittest.TestCase):
    """Test trace parsing and execution."""

    def setUp(self):
        """Set up a fresh symbolic state for each test."""
        self.state = SymbolicState()

    def test_mov_immediate_to_register(self):
        """Test mov $imm, %reg instructions."""
        trace = ["mov $42, %rax"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        self.assertEqual(self.state.reg_versions["rax"], 1)
        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), "42")

    def test_mov_immediate_to_memory(self):
        """Test mov $imm, mem instructions."""
        trace = ["mov $100, (%rsp)"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Should create memory variable
        addr_str = str(BinOp("+", Var("rsp", 0), Const(0)))
        self.assertIn(addr_str, self.state.mem)
        self.assertEqual(str(self.state.mem[addr_str]), "100")

    def test_mov_memory_to_register(self):
        """Test mov mem, %reg instructions."""
        # First store something in memory
        trace1 = ["mov $55, (%rsp)"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace1)

        # Then load it to register
        trace2 = ["mov (%rsp), %rax"]
        engine.parse_trace_and_execute(trace2)

        self.assertIn("rax_1", self.state.definitions)

    def test_add_immediate_to_register(self):
        """Test add $imm, %reg instructions."""
        trace = [
            "mov $10, %rax",
            "add $5, %rax"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Should have rax_1 = 10, rax_2 = (rax_1 + 5)
        print(self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), "10")
        self.assertIsInstance(self.state.definitions["rax_2"], BinOp)

    def test_add_register_to_memory(self):
        """Test add %reg, mem instructions."""
        trace = [
            "mov $20, %rax",
            "mov $30, (%rsp)",
            "add %rax, (%rsp)"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Memory should be updated with addition
        addr_str = str(BinOp("+", Var("rsp", 0), Const(0)))
        self.assertIn(addr_str, self.state.mem)

    def test_add_memory_to_register(self):
        """Test add mem, %reg instructions."""
        trace = [
            "mov $15, %rax",
            "mov $25, (%rsp)",
            "add (%rsp), %rax"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Register should be updated
        self.assertIn("rax_2", self.state.definitions)
        self.assertIsInstance(self.state.definitions["rax_2"], BinOp)

    def test_skipped_instructions(self):
        """Test that certain instructions are properly skipped."""
        trace = [
            "endbr64",
            "nop",
            "cmp %rax, %rbx",
            "jne label",
            "mov $42, %rax",
            "ret"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Only the mov should be executed
        self.assertEqual(len(self.state.reg_versions), 1)  # Only rax
        self.assertEqual(str(self.state.definitions["rax_1"]), "42")

    def test_register_name_normalization(self):
        """Test that %e* registers are normalized to %r*."""
        trace = ["mov $99, %eax"]  # Should become %rax
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), "99")


class TestExpressionExpansion(unittest.TestCase):
    """Test expression expansion and optimization."""

    def setUp(self):
        """Set up symbolic state with some definitions."""
        self.state = SymbolicState()
        self.state.write_reg("rax", Const(42))
        self.state.write_reg("rbx", BinOp("+", Var("rax", 1), Const(8)))

    def test_expand_const(self):
        """Test expanding constant expressions."""
        expr = Const(100)
        expanded = expand_expr(expr, self.state)
        self.assertEqual(expanded, expr)

    def test_expand_var_with_definition(self):
        """Test expanding variables that have definitions."""
        var = Var("rax", 1)
        expanded = expand_expr(var, self.state)
        self.assertIsInstance(expanded, Const)
        self.assertEqual(expanded.value, 42)

    def test_expand_var_without_definition(self):
        """Test expanding variables without definitions."""
        var = Var("rcx", 1)  # No definition for this
        expanded = expand_expr(var, self.state)
        self.assertEqual(expanded, var)

    def test_expand_binop(self):
        """Test expanding binary operations."""
        var = Var("rbx", 1)  # rbx_2 = (rax_1 + 8)
        expanded = expand_expr(var, self.state)
        # Should expand to (42 + 8)
        self.assertIsInstance(expanded, BinOp)
        self.assertEqual(expanded.op, "+")

    def test_expand_mem(self):
        """Test expanding memory expressions."""
        # Store something in memory first
        addr = Const(0x1000)
        self.state.mem_store(addr, Const(123))

        mem_expr = Mem(addr)
        expanded = expand_expr(mem_expr, self.state)
        self.assertIsInstance(expanded, Const)
        self.assertEqual(expanded.value, 123)

    def test_expand_mem_without_value(self):
        """Test expanding memory expressions without stored values."""
        addr = Const(0x2000)
        mem_expr = Mem(addr)
        expanded = expand_expr(mem_expr, self.state)
        self.assertIsInstance(expanded, Mem)


class TestExpressionOptimization(unittest.TestCase):
    """Test expression optimization and simplification."""

    def setUp(self):
        """Set up symbolic state."""
        self.state = SymbolicState()

    def test_constant_folding_addition(self):
        """Test constant folding for addition."""
        expr = BinOp("+", Const(10), Const(5))
        optimized = optimize_expr(expr, self.state)
        self.assertIsInstance(optimized, Const)
        self.assertEqual(optimized.value, 15)

    def test_constant_folding_subtraction(self):
        """Test constant folding for subtraction."""
        expr = BinOp("-", Const(20), Const(7))
        optimized = optimize_expr(expr, self.state)
        self.assertIsInstance(optimized, Const)
        self.assertEqual(optimized.value, 13)

    def test_constant_folding_multiplication(self):
        """Test constant folding for multiplication."""
        expr = BinOp("*", Const(6), Const(7))
        optimized = optimize_expr(expr, self.state)
        self.assertIsInstance(optimized, Const)
        self.assertEqual(optimized.value, 42)

    def test_constant_folding_division(self):
        """Test constant folding for division."""
        expr = BinOp("/", Const(20), Const(4))
        optimized = optimize_expr(expr, self.state)
        self.assertIsInstance(optimized, Const)
        self.assertEqual(optimized.value, 5)

    def test_addition_identity_left(self):
        """Test x + 0 = x optimization."""
        var = Var("rax", 1)
        expr = BinOp("+", Const(0), var)
        optimized = optimize_expr(expr, self.state)
        self.assertEqual(optimized, var)

    def test_addition_identity_right(self):
        """Test 0 + x = x optimization."""
        var = Var("rax", 1)
        expr = BinOp("+", var, Const(0))
        optimized = optimize_expr(expr, self.state)
        self.assertEqual(optimized, var)

    def test_subtraction_identity(self):
        """Test x - 0 = x optimization."""
        var = Var("rax", 1)
        expr = BinOp("-", var, Const(0))
        optimized = optimize_expr(expr, self.state)
        self.assertEqual(optimized, var)

    def test_multiplication_identity(self):
        """Test x * 1 = x and 1 * x = x optimization."""
        var = Var("rax", 1)

        expr1 = BinOp("*", var, Const(1))
        optimized1 = optimize_expr(expr1, self.state)
        self.assertEqual(optimized1, var)

        expr2 = BinOp("*", Const(1), var)
        optimized2 = optimize_expr(expr2, self.state)
        self.assertEqual(optimized2, var)

    def test_multiplication_zero(self):
        """Test x * 0 = 0 and 0 * x = 0 optimization."""
        var = Var("rax", 1)

        expr1 = BinOp("*", var, Const(0))
        optimized1 = optimize_expr(expr1, self.state)
        self.assertIsInstance(optimized1, Const)
        self.assertEqual(optimized1.value, 0)

        expr2 = BinOp("*", Const(0), var)
        optimized2 = optimize_expr(expr2, self.state)
        self.assertIsInstance(optimized2, Const)
        self.assertEqual(optimized2.value, 0)

    def test_complex_constant_collection(self):
        """Test collection of constants in addition chains."""
        # (x + 5) + 10 should become x + 15
        var = Var("rax", 1)
        inner = BinOp("+", var, Const(5))
        outer = BinOp("+", inner, Const(10))

        optimized = optimize_expr(outer, self.state)
        # Result structure may vary, but should contain the variable and 15
        opt_str = str(optimized)
        self.assertIn("rax_1", opt_str)
        self.assertIn("15", opt_str)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_complete_execution_flow(self):
        """Test a complete execution flow with optimization."""
        state = SymbolicState()

        trace = [
            "mov $10, %rax",
            "mov $20, %rbx",
            "add %rbx, %rax",
            "mov %rax, (%rsp)",
            "add $5, (%rsp)"
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        # Check final state
        self.assertIn("rax_2", state.definitions)  # Final rax version

        # Check memory was updated
        addr_str = str(BinOp("+", Var("rsp", 0), Const(0)))
        self.assertIn(addr_str, state.mem)

        # Test optimization of final expression
        final_rax = Var("rax", 2)
        optimized = optimize_expr(final_rax, state)
        # Should resolve to some expression involving the operations

    def test_fibonacci_like_sequence(self):
        """Test a fibonacci-like sequence of operations."""
        state = SymbolicState()

        trace = [
            "mov $1, %rax",      # rax = 1
            "mov $1, %rbx",      # rbx = 1
            "add %rbx, %rax",    # rax = rax + rbx = 2
            "add %rax, %rbx",    # rbx = rbx + rax = 3
            "add %rbx, %rax",    # rax = rax + rbx = 5
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        # Expand and optimize final values
        final_rax = state.current_var("rax")
        final_rbx = state.current_var("rbx")

        expanded_rax = expand_expr(final_rax, state)
        expanded_rbx = expand_expr(final_rbx, state)

        # These should be complex expressions
        self.assertIsInstance(expanded_rax, BinOp)
        self.assertIsInstance(expanded_rbx, BinOp)

    def test_memory_aliasing(self):
        """Test memory operations with same addresses."""
        state = SymbolicState()

        trace = [
            "mov $100, (%rsp)",
            "add $50, (%rsp)",
            "mov (%rsp), %rax"
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        # The memory should contain the updated value
        addr_str = str(BinOp("+", Var("rsp", 0), Const(0)))
        self.assertIn(addr_str, state.mem)

        # The register should load from memory
        final_rax = state.current_var("rax")
        expanded = expand_expr(final_rax, state)
        # Should be some expression representing the memory load

    def test_file_based_trace(self):
        """Test execution from a temporary trace file."""
        # Create a temporary trace file
        trace_content = """
400500: mov $42, %rax
400505: mov $8, %rbx  
40050a: add %rbx, %rax
40050f: mov %rax, (%rsp)
        """.strip()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(trace_content)
            temp_file = f.name

        try:
            # Read and process the trace like the main function does
            with open(temp_file, 'r') as f:
                trace = f.readlines()
            trace = [line.strip() for line in trace if line.strip()]
            trace = [line.split(':')[1].strip()
                     for line in trace if ':' in line]

            state = SymbolicState()
            engine = SymbolicEngine(state)
            engine.parse_trace_and_execute(trace)

            # Verify execution
            self.assertIn("rax_2", state.definitions)
            self.assertGreater(len(state.mem), 0)

        finally:
            os.unlink(temp_file)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_instruction_format(self):
        """Test handling of invalid instruction formats."""
        state = SymbolicState()
        trace = ["invalid instruction format"]

        # Should not crash, just log warnings
        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)
        self.assertEqual(len(state.reg_versions), 0)

    def test_malformed_address(self):
        """Test handling of malformed addresses."""
        state = SymbolicState()

        with self.assertRaises(ValueError):
            parse_address("malformed", state)

    def test_empty_trace(self):
        """Test execution of empty trace."""
        state = SymbolicState()
        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute([])

        # Should not crash and state should be minimal
        self.assertEqual(len(state.reg_versions), 0)
        self.assertEqual(len(state.mem), 0)

    def test_whitespace_handling(self):
        """Test handling of whitespace in traces."""
        state = SymbolicState()
        trace = [
            "   mov $42, %rax   ",
            "\t\tadd $8, %rax\t",
            "",
            "   ",
            "mov %rax, (%rsp)"
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        self.assertIn("rax_2", state.definitions)


if __name__ == '__main__':
    unittest.main(verbosity=2)
