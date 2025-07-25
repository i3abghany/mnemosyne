#!/usr/bin/env python3

from engine import (
    Expr, Const, Var, Mem, BinOp, SymbolicState,
    SymbolicEngine, expand_expr, optimize_expr
)
import unittest
import tempfile
import os
import sys

import keystone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExpressions(unittest.TestCase):

    def test_const_creation_and_str(self):
        c1 = Const(42)
        self.assertEqual(c1.value, 42)
        self.assertEqual(str(c1), "42")

        c2 = Const(-10)
        self.assertEqual(str(c2), "-10")

        c3 = Const(0)
        self.assertEqual(str(c3), "0")

    def test_var_creation_and_str(self):
        v1 = Var("rax", 1)
        self.assertEqual(v1.name, "rax")
        self.assertEqual(v1.version, 1)
        self.assertEqual(str(v1), "rax_1")

        v2 = Var("rbx", 10)
        self.assertEqual(str(v2), "rbx_10")

    def test_mem_creation_and_str(self):
        addr = Const(4096)
        m1 = Mem(addr)
        self.assertEqual(str(m1), "mem[4096]")

        addr2 = BinOp("+", Var("rsp", 1), Const(8))
        m2 = Mem(addr2)
        self.assertEqual(str(m2), "mem[(rsp_1 + 8)]")

    def test_binop_creation_and_str(self):
        left = Var("rax", 1)
        right = Const(42)

        add = BinOp("+", left, right)
        self.assertEqual(str(add), "(rax_1 + 42)")

        sub = BinOp("-", left, right)
        self.assertEqual(str(sub), "(rax_1 - 42)")

        mul = BinOp("*", left, right)
        self.assertEqual(str(mul), "(rax_1 * 42)")

    def test_nested_binop(self):
        inner = BinOp("*", Var("rbx", 2), Const(4))
        outer = BinOp("+", Var("rax", 1), inner)
        self.assertEqual(str(outer), "(rax_1 + (rbx_2 * 4))")


class TestSymbolicState(unittest.TestCase):

    def setUp(self):
        self.state = SymbolicState()

    def test_initial_state(self):
        self.assertEqual(self.state.reg_versions, {})
        self.assertEqual(len(self.state.mem), 0)
        self.assertIn("rsp_1", self.state.definitions)
        self.assertEqual(str(self.state.rsp_val), "4096")  # 0x1000

    def test_new_var_creation(self):
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
        # Before any writes
        v1 = self.state.current_var("rax")
        self.assertEqual(str(v1), "rax_0")

        # After creating a new version
        self.state.new_var("rax")
        v2 = self.state.current_var("rax")
        self.assertEqual(str(v2), "rax_1")

    def test_write_reg(self):
        expr = Const(42)
        var, returned_expr = self.state.write_reg("rax", expr)

        self.assertEqual(str(var), "rax_1")
        self.assertEqual(returned_expr, expr)
        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(self.state.definitions["rax_1"], expr)

    def test_read_reg(self):
        """Test register reads."""
        self.state.write_reg("rax", Const(42))
        var = self.state.read_reg("rax")
        self.assertEqual(str(var), "rax_1")

    def test_mem_store_and_load(self):
        """Test memory operations."""
        addr = BinOp("+", Var("rsp", 0), Const(8))
        value = Var(name='mem[(rsp_0 + 8)]', version=1)

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


@unittest.skip("Legacy tests, not used in current implementation")
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
        trace = ["movq $42, %rax"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        self.assertEqual(self.state.reg_versions["rax"], 1)
        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), str(0x42))

    def test_mov_immediate_to_memory(self):
        """Test mov $imm, mem instructions."""
        trace = ["movq $100, (%rsp)"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Should create memory variable
        addr_str = str(Var("rsp", 0))
        self.assertIn(addr_str, self.state.mem)
        self.assertEqual(str(self.state.mem[addr_str]), str(0x100))

    def test_mov_memory_to_register(self):
        """Test mov mem, %reg instructions."""
        # First store something in memory
        trace1 = ["movq $55, (%rsp)"]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace1)

        # Then load it to register
        trace2 = ["movq (%rsp), %rax"]
        engine.parse_trace_and_execute(trace2)

        self.assertIn("rax_1", self.state.definitions)

    def test_add_immediate_to_register(self):
        """Test add $imm, %reg instructions."""
        trace = [
            "movq $10, %rax",
            "addq $5, %rax"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Should have rax_1 = 10, rax_2 = (rax_1 + 5)
        print(self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), str(0x10))
        self.assertIsInstance(self.state.definitions["rax_2"], BinOp)

    def test_add_register_to_memory(self):
        """Test add %reg, mem instructions."""
        trace = [
            "movq $20, %rax",
            "movq $30, (%rsp)",
            "addq %rax, (%rsp)"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Memory should be updated with addition
        addr_str = str(Var("rsp", 0))
        self.assertIn(addr_str, self.state.mem)

    def test_add_memory_to_register(self):
        """Test add mem, %reg instructions."""
        trace = [
            "movq $15, %rax",
            "movq $25, (%rsp)",
            "addq (%rsp), %rax"
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
            "jne 0x1000",
            "movq $42, %rax",
            "retq"
        ]
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        # Only the mov should be executed
        self.assertEqual(len(self.state.reg_versions), 1)  # Only rax
        self.assertEqual(str(self.state.definitions["rax_1"]), str(0x42))

    def test_register_name_normalization(self):
        """Test that %e* registers are normalized to %r*."""
        trace = ["mov $99, %eax"]  # Should become %rax
        engine = SymbolicEngine(self.state)
        engine.parse_trace_and_execute(trace)

        self.assertIn("rax_1", self.state.definitions)
        self.assertEqual(str(self.state.definitions["rax_1"]), str(0x99))


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

    def test_expand_cyclic_memory_reference(self):
        """Test expanding memory expressions that reference themselves (cyclic references)."""
        addr = Var("rsp", 0)  # rsp_0
        mem_expr = Mem(addr)

        cyclic_value = BinOp("+", mem_expr, Const(1))  # (mem[rsp_0] + 1)
        self.state.mem_store(addr, cyclic_value)

        expanded = expand_expr(mem_expr, self.state)
        self.assertIsInstance(expanded, Mem)
        self.assertEqual(str(expanded), "mem[rsp_0]")

    def test_expand_complex_cyclic_memory_chain(self):
        """Test expanding indirect cyclic memory references."""
        addr1 = Const(0x1000)
        addr2 = Const(0x2000)

        mem1 = Mem(addr1)
        mem2 = Mem(addr2)

        # mem[0x1000] = (mem[0x2000] + 1)
        self.state.mem_store(addr1, BinOp("+", mem2, Const(1)))
        # mem[0x2000] = (mem[0x1000] + 2)
        self.state.mem_store(addr2, BinOp("+", mem1, Const(2)))

        expanded1 = expand_expr(mem1, self.state)
        expanded2 = expand_expr(mem2, self.state)

        self.assertIsNotNone(expanded1)
        self.assertIsNotNone(expanded2)

    def test_expand_non_cyclic_memory_with_memory_references(self):
        """Test that non-cyclic memory references still expand correctly."""
        addr1 = Const(0x1000)
        addr2 = Const(0x2000)

        self.state.mem_store(addr1, Const(100))
        self.state.mem_store(addr2, BinOp("+", Mem(addr1), Const(50)))

        mem2_expr = Mem(addr2)
        expanded = expand_expr(mem2_expr, self.state)

        self.assertIsInstance(expanded, BinOp)
        self.assertEqual(expanded.op, "+")
        self.assertIsInstance(expanded.left, Const)
        self.assertEqual(expanded.left.value, 100)
        self.assertIsInstance(expanded.right, Const)
        self.assertEqual(expanded.right.value, 50)

    def test_expand_memory_with_symbolic_address(self):
        rsp_var = Var("rsp", 0)  # rsp_0
        mem_rsp = Mem(rsp_var)   # mem[rsp_0]

        cyclic_expr = BinOp("+", mem_rsp, Const(1))
        self.state.mem_store(rsp_var, cyclic_expr)

        expanded = expand_expr(mem_rsp, self.state)
        self.assertIsInstance(expanded, Mem)
        self.assertEqual(str(expanded.addr), "rsp_0")

    def test_expand_deeply_nested_non_cyclic(self):
        addr1 = Const(0x1000)
        addr2 = Const(0x2000)
        addr3 = Const(0x3000)

        self.state.mem_store(addr1, Const(5))
        self.state.mem_store(addr2, BinOp("+", Mem(addr1), Const(1)))
        self.state.mem_store(addr3, BinOp("*", Mem(addr2), Const(2)))

        expanded = expand_expr(Mem(addr3), self.state)
        self.assertIsInstance(expanded, BinOp)
        self.assertEqual(expanded.op, "*")

        self.assertIsInstance(expanded.left, BinOp)
        self.assertEqual(expanded.left.op, "+")


class TestExpressionOptimization(unittest.TestCase):

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
        var = Var("rax", 1)
        inner = BinOp("+", var, Const(5))
        outer = BinOp("+", inner, Const(10))

        optimized = optimize_expr(outer, self.state)
        opt_str = str(optimized)
        self.assertIn("rax_1", opt_str)
        self.assertIn("15", opt_str)


class TestIntegration(unittest.TestCase):
    def test_complete_execution_flow(self):
        state = SymbolicState()

        trace = [
            "mov $10, %rax",
            "mov $20, %rbx",
            "add %rbx, %rax",
            "mov %rax, (%rsp)",
            # "5" and "(%rsp)" are not enough to determine the operation size. suffix "q" is needed
            "addq $5, (%rsp)"
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        # Check final state
        self.assertIn("rax_2", state.definitions)  # Final rax version

        # Check memory was updated
        addr_str = str(Var("rsp", 0))
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
        state = SymbolicState()

        trace = [
            "movq $100, (%rsp)",
            "addq $50, (%rsp)",
            "mov (%rsp), %rax"
        ]

        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute(trace)

        addr_str = str(Var("rsp", 0))
        self.assertIn(addr_str, state.mem)

        final_rax = state.current_var("rax")
        expanded = expand_expr(final_rax, state)
        optimized = optimize_expr(expanded, state)
        self.assertIsInstance(optimized, Const)
        self.assertEqual(optimized.value, 0x100 + 0x50)

    def test_file_based_trace(self):
        trace_content = """
0x400500: mov $42, %rax
0x400505: mov $8, %rbx  
0x40050a: add %rbx, %rax
0x40050f: mov %rax, (%rsp)
        """.strip()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(trace_content)
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                trace = f.readlines()
            trace = [line.strip() for line in trace if line.strip()]
            trace = [line.split(':')[1].strip()
                     for line in trace if ':' in line]

            state = SymbolicState()
            engine = SymbolicEngine(state)
            engine.parse_trace_and_execute(trace)

            self.assertIn("rax_2", state.definitions)
            self.assertGreater(len(state.mem), 0)

        finally:
            os.unlink(temp_file)

    def test_cyclic_memory_trace_execution(self):
        state = SymbolicState()
        engine = SymbolicEngine(state)

        trace = [
            # mem[rsp_0] = (mem[rsp_0] + 1) - creates cycle
            "addq $1, (%rsp)",
            "mov (%rsp), %rax",   # rax_1 = (mem[rsp_0] + 1)
            "mov (%rsp), %rax",   # rax_2 = (mem[rsp_0] + 1)
        ]

        # This should NOT cause infinite recursion
        engine.parse_trace_and_execute(trace)

        final_rax = state.current_var("rax")
        self.assertEqual(str(final_rax), "rax_2")

        expanded = expand_expr(final_rax, state)
        self.assertIsInstance(expanded, BinOp)
        self.assertEqual(expanded.op, "+")
        self.assertIsInstance(expanded.left, Mem)
        self.assertEqual(str(expanded.left), "mem[rsp_0]")
        self.assertIsInstance(expanded.right, Const)
        self.assertEqual(expanded.right.value, 1)

        optimized = optimize_expr(expanded, state)
        self.assertIsInstance(optimized, BinOp)

    def test_multiple_cyclic_memory_operations(self):
        state = SymbolicState()
        engine = SymbolicEngine(state)

        trace = [
            "movq $10, (%rsp)",      # mem[rsp_0] = 10
            "addq $5, (%rsp)",       # mem[rsp_0] = (mem[rsp_0] + 5) = (10 + 5)
            # rax_1 = (rax_0 + mem[rsp_0]) = (rax_0 + (10 + 5))
            "addq (%rsp), %rax",
            "movq %rax, 8(%rsp)",    # mem[(8 + rsp_0)] = rax_1
            "movq 8(%rsp), %rbx",    # rbx_1 = rax_1 - load stored value
            # rax_2 = (rax_1 + rbx_1) - more complex expression
            "addq %rbx, %rax",
        ]

        engine.parse_trace_and_execute(trace)

        self.assertIn("rax_1", state.definitions)
        self.assertGreater(len(state.mem), 1)

        final_rax = state.current_var("rax")
        expanded_rax = expand_expr(final_rax, state)
        self.assertIsNotNone(expanded_rax)


class TestErrorHandling(unittest.TestCase):
    def test_invalid_instruction_format(self):
        state = SymbolicState()
        trace = ["invalid instruction format"]

        engine = SymbolicEngine(state)
        with self.assertRaises(keystone.keystone.KsError):
            engine.parse_trace_and_execute(trace)

    def test_empty_trace(self):

        state = SymbolicState()
        engine = SymbolicEngine(state)
        engine.parse_trace_and_execute([])

        self.assertEqual(len(state.reg_versions), 0)
        self.assertEqual(len(state.mem), 0)

    def test_whitespace_handling(self):
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
