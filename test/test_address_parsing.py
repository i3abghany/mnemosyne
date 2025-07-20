#!/usr/bin/env python3

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import SymbolicState, parse_address, Var, Mem, BinOp, Const


class TestAddressParsing(unittest.TestCase):
    """Test address parsing functionality with comprehensive coverage."""

    def setUp(self):
        """Set up a fresh symbolic state and initialize common registers."""
        self.state = SymbolicState()

        # Pre-initialize registers to known versioned variables for consistent testing
        common_regs = ["rax", "rbx", "rcx", "rdx",
                       "rsi", "rdi", "rsp", "rbp", "r8", "r9"]
        for reg in common_regs:
            self.state.write_reg(reg, Const(0))  # creates e.g., rax_1 = 0

    def test_simple_register_indirect(self):
        """Test simple register indirect addressing: (%reg)."""
        test_cases = [
            ("(%rax)", "(rax_1 + 0)"),
            ("(%rbx)", "(rbx_1 + 0)"),
            ("(%rcx)", "(rcx_1 + 0)"),
            ("(%rsp)", "(rsp_1 + 0)"),
            ("(%rbp)", "(rbp_1 + 0)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_positive_displacement(self):
        """Test positive displacement addressing."""
        test_cases = [
            ("8(%rax)", "(rax_1 + 8)"),
            ("16(%rbx)", "(rbx_1 + 16)"),
            ("32(%rcx)", "(rcx_1 + 32)"),
            ("64(%rsp)", "(rsp_1 + 64)"),
            ("128(%rbp)", "(rbp_1 + 128)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_negative_displacement(self):
        """Test negative displacement addressing."""
        test_cases = [
            ("-8(%rax)", "(rax_1 + -8)"),
            ("-16(%rbx)", "(rbx_1 + -16)"),
            ("-32(%rcx)", "(rcx_1 + -32)"),
            ("-64(%rsp)", "(rsp_1 + -64)"),
            ("-128(%rbp)", "(rbp_1 + -128)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_hex_displacement(self):
        """Test hexadecimal displacement addressing."""
        test_cases = [
            ("0x8(%rax)", "(rax_1 + 8)"),
            ("0x10(%rbx)", "(rbx_1 + 16)"),
            ("0x20(%rcx)", "(rcx_1 + 32)"),
            ("0xff(%rdx)", "(rdx_1 + 255)"),
            ("0x100(%rsi)", "(rsi_1 + 256)"),
            ("-0x8(%rsp)", "(rsp_1 + -8)"),
            ("-0x10(%rbp)", "(rbp_1 + -16)"),
            ("-0x28(%rdi)", "(rdi_1 + -40)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_indexed_addressing_no_scale(self):
        """Test indexed addressing without explicit scale (scale=1)."""
        test_cases = [
            ("(%rax,%rbx)", "(rax_1 + rbx_1)"),
            ("(%rcx,%rdx)", "(rcx_1 + rdx_1)"),
            ("(%rsi,%rdi)", "(rsi_1 + rdi_1)"),
            ("(%rsp,%rax)", "(rsp_1 + rax_1)"),
            ("(%rbp,%rcx)", "(rbp_1 + rcx_1)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_indexed_addressing_with_scale(self):
        """Test indexed addressing with scale factors."""
        test_cases = [
            ("(%rax,%rbx,1)", "(rax_1 + (rbx_1 * 1))"),
            ("(%rax,%rbx,2)", "(rax_1 + (rbx_1 * 2))"),
            ("(%rax,%rbx,4)", "(rax_1 + (rbx_1 * 4))"),
            ("(%rax,%rbx,8)", "(rax_1 + (rbx_1 * 8))"),
            ("(%rcx,%rdx,4)", "(rcx_1 + (rdx_1 * 4))"),
            ("(%rsi,%rdi,2)", "(rsi_1 + (rdi_1 * 2))"),
            ("(%rsp,%rax,8)", "(rsp_1 + (rax_1 * 8))"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_displacement_with_indexed_addressing(self):
        """Test displacement combined with indexed addressing."""
        test_cases = [
            ("8(%rax,%rbx)", "((rax_1 + rbx_1) + 8)"),
            ("16(%rcx,%rdx,2)", "((rcx_1 + (rdx_1 * 2)) + 16)"),
            ("32(%rsi,%rdi,4)", "((rsi_1 + (rdi_1 * 4)) + 32)"),
            ("-8(%rsp,%rax)", "((rsp_1 + rax_1) + -8)"),
            ("-16(%rbp,%rcx,2)", "((rbp_1 + (rcx_1 * 2)) + -16)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_hex_displacement_with_indexed_addressing(self):
        """Test hexadecimal displacement with indexed addressing."""
        test_cases = [
            ("0x8(%rax,%rbx)", "((rax_1 + rbx_1) + 8)"),
            ("0x10(%rcx,%rdx,2)", "((rcx_1 + (rdx_1 * 2)) + 16)"),
            ("0x20(%rsi,%rdi,4)", "((rsi_1 + (rdi_1 * 4)) + 32)"),
            ("-0x8(%rsp,%rax)", "((rsp_1 + rax_1) + -8)"),
            ("-0x10(%rbp,%rcx,2)", "((rbp_1 + (rcx_1 * 2)) + -16)"),
            ("-0x28(%rsp,%rax,4)", "((rsp_1 + (rax_1 * 4)) + -40)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_index_only_addressing(self):
        """Test addressing with index register but no base (rare but valid)."""

        test_cases = [
            ("(,%rax,2)", "(rax_1 * 2)"),
            ("(,%rbx,4)", "(rbx_1 * 4)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_whitespace_handling(self):
        """Test address parsing with various whitespace patterns."""
        test_cases = [
            ("(%rax, %rbx)", "(rax_1 + rbx_1)"),
            ("( %rax , %rbx )", "(rax_1 + rbx_1)"),
            ("(%rax,%rbx, 2)", "(rax_1 + (rbx_1 * 2))"),
            ("( %rax , %rbx , 4 )", "(rax_1 + (rbx_1 * 4))"),
            ("8( %rax , %rbx )", "((rax_1 + rbx_1) + 8)"),
            ("16( %rcx , %rdx , 2 )", "((rcx_1 + (rdx_1 * 2)) + 16)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_complex_addressing_modes(self):
        """Test complex real-world addressing patterns."""
        test_cases = [
            # Array access patterns
            ("(%rsi,%rdi,4)", "(rsi_1 + (rdi_1 * 4))"),
            ("(%rax,%rbx,8)", "(rax_1 + (rbx_1 * 8))"),

            # Stack frame access patterns
            ("-8(%rbp)", "(rbp_1 + -8)"),
            ("-16(%rbp)", "(rbp_1 + -16)"),

            # parameter or saved register
            ("8(%rsp)", "(rsp_1 + 8)"),

            # Structure field access patterns
            ("0x10(%rax)", "(rax_1 + 16)"),
            ("0x20(%rbx,%rcx,1)", "((rbx_1 + (rcx_1 * 1)) + 32)"),

            # Complex stack access
            ("-0x28(%rsp,%rax,4)", "((rsp_1 + (rax_1 * 4)) + -40)"),
            ("0x100(%rsi,%rdi,8)", "((rsi_1 + (rdi_1 * 8)) + 256)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_register_name_variations(self):
        """Test different register name formats."""
        # Initialize some r8-r15 registers for testing (but not the common ones already initialized)
        r_regs = {}
        for i in range(8, 16):
            var, _ = self.state.write_reg(f"r{i}", Const(0))
            r_regs[f"r{i}"] = var.version

        test_cases = [
            ("(%r8)", f"(r8_{r_regs['r8']} + 0)"),
            ("(%r9)", f"(r9_{r_regs['r9']} + 0)"),
            ("8(%r10)", f"(r10_{r_regs['r10']} + 8)"),
            ("(%rax,%r11,2)", f"(rax_1 + (r11_{r_regs['r11']} * 2))"),
            ("-0x8(%r12,%r13,4)",
             f"((r12_{r_regs['r12']} + (r13_{r_regs['r13']} * 4)) + -8)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_large_displacements(self):
        """Test parsing with large displacement values."""
        test_cases = [
            ("0x1000(%rax)", "(rax_1 + 4096)"),
            ("0xffff(%rbx)", "(rbx_1 + 65535)"),
            ("-0x1000(%rcx)", "(rcx_1 + -4096)"),
            ("2147483647(%rdx)", "(rdx_1 + 2147483647)"),
            ("-2147483648(%rsi)", "(rsi_1 + -2147483648)"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_error_cases(self):
        """Test error handling for invalid address formats."""
        invalid_cases = [
            "",
            "()",
            "(%invalid)",
            "(%rax,%rbx,3)",
            "(%rax,%rbx,16)",
            "invalid_format",
        ]

        for invalid_addr in invalid_cases:
            with self.subTest(addr=invalid_addr):
                with self.assertRaises(ValueError):
                    parse_address(invalid_addr, self.state)

    def test_edge_case_scales(self):
        """Test all valid scale factors."""
        test_cases = [
            ("(%rax,%rbx,1)", "(rax_1 + (rbx_1 * 1))"),
            ("(%rax,%rbx,2)", "(rax_1 + (rbx_1 * 2))"),
            ("(%rax,%rbx,4)", "(rax_1 + (rbx_1 * 4))"),
            ("(%rax,%rbx,8)", "(rax_1 + (rbx_1 * 8))"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_expression_consistency(self):
        """Test that parsing creates consistent expression trees."""
        addr_str = "0x10(%rax,%rbx,2)"
        result = parse_address(addr_str, self.state)

        # Check that the result is a BinOp (addition)
        self.assertIsInstance(result, BinOp)
        self.assertEqual(result.op, "+")

    def test_zero_values(self):
        """Test addressing with zero displacements and scales."""
        test_cases = [
            ("0(%rax)", "(rax_1 + 0)"),
            ("0x0(%rbx)", "(rbx_1 + 0)"),
            ("(%rax,%rbx,1)", "(rax_1 + (rbx_1 * 1))"),
        ]

        for addr_str, expected in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertEqual(str(result), expected)

    def test_address_expression_types(self):
        """Test that parsed addresses return correct expression types."""
        test_cases = [
            "(%rax)",
            "8(%rbx)",
            "(%rcx,%rdx,2)",
            "0x10(%rsi,%rdi,4)",
        ]

        for addr_str in test_cases:
            with self.subTest(addr=addr_str):
                result = parse_address(addr_str, self.state)
                self.assertIsInstance(result, BinOp)
                self.assertEqual(result.op, "+")


class TestAddressParsingIntegration(unittest.TestCase):
    """Integration tests for address parsing with symbolic state."""

    def setUp(self):
        """Set up symbolic state with realistic register values."""
        self.state = SymbolicState()

        # Initialize registers with more realistic symbolic values
        self.state.write_reg("rsp", BinOp("+", Const(0x1000), Const(-16)))
        self.state.write_reg("rbp", BinOp("+", Const(0x1000), Const(0)))
        self.state.write_reg("rax", Var("input", 1))
        self.state.write_reg("rbx", BinOp("*", Var("index", 1), Const(4)))

    def test_realistic_stack_access(self):
        """Test parsing addresses that would be used in real stack access."""
        addr1 = parse_address("-8(%rbp)", self.state)
        self.assertIsInstance(addr1, BinOp)

        addr2 = parse_address("8(%rsp)", self.state)
        self.assertIsInstance(addr2, BinOp)

        addr3 = parse_address("(%rax,%rbx,1)", self.state)
        self.assertIsInstance(addr3, BinOp)

    def test_address_with_symbolic_registers(self):
        """Test that addresses work correctly with symbolic register values."""
        addr_str = "(%rax,%rbx,2)"
        result = parse_address(addr_str, self.state)

        self.assertIsInstance(result, BinOp)
        result_str = str(result)

        self.assertIn("rax_", result_str)
        self.assertIn("rbx_", result_str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
