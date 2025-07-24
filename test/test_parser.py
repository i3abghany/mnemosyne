#!/usr/bin/env python3

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from keystone import *
    from capstone import *
    from capstone.x86_const import *
    KEYSTONE_AVAILABLE = True
except ImportError:
    KEYSTONE_AVAILABLE = False

if KEYSTONE_AVAILABLE:
    from parser import TraceParser


@unittest.skipUnless(KEYSTONE_AVAILABLE, "Keystone/Capstone not available")
class TestTraceParser(unittest.TestCase):
    def setUp(self):
        self.simple_trace = ["mov %rax, %rbx"]
        self.parser = TraceParser(self.simple_trace)

    def test_init_basic(self):
        trace = ["mov %rax, %rbx", "add $1, %rcx"]
        parser = TraceParser(trace)
        self.assertEqual(parser.trace, trace)
        self.assertIsNotNone(parser.ks)
        self.assertIsNotNone(parser.md)

    def test_init_empty_trace(self):
        parser = TraceParser([])
        self.assertEqual(parser.trace, [])
        self.assertIsNotNone(parser.ks)
        self.assertIsNotNone(parser.md)

    def test_parse_single_instruction(self):
        trace = ["movq %rax, %rbx"]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].mnemonic, trace[0].split()[0])
        self.assertEqual(len(result[0].operands), 2)
        self.assertIsInstance(result[0].address, int)
        self.assertIsInstance(result[0].size, int)

    def test_parse_multiple_instructions(self):
        trace = [
            "movq %rax, %rbx",
            "addq $10, %rcx",
            "subq %rdx, %rsi"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 3)
        mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(mnemonics, [line.split()[0] for line in trace])

    def test_parse_arithmetic_instructions(self):
        trace = [
            "addq %rax, %rbx",
            "subq %rcx, %rdx",
            "mulq %rsi",
            "divq %rdi",
            "incq %r8",
            "decq %r9"
        ]
        parser = TraceParser(trace)
        result = parser.parse()

        expected_mnemonics = [line.split()[0] for line in trace]
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_logical_instructions(self):
        trace = [
            "andq %rax, %rbx",
            "orq %rcx, %rdx",
            "xorq %rsi, %rdi",
            "notq %r8",
            "shlq $2, %r9",
            "shrq $1, %r10"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = [line.split()[0] for line in trace]
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_memory_instructions(self):
        trace = [
            "movq %rax, (%rbx)",
            "movq (%rcx), %rdx",
            "movq $100, 8(%rsi)",
            "leaq 16(%rdi), %r8",
            "pushq %r9",
            "popq %r10"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        expected_mnemonics = [line.split()[0] for line in trace]
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_control_flow_instructions(self):
        trace = [
            "callq *%rax",
            "retq",
            "jmpq *%rcx",
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        control_mnemonics = [insn.mnemonic for insn in result]
        expected_mnemonics = ['callq', 'retq', 'jmpq']
        self.assertEqual(control_mnemonics, expected_mnemonics)

    def test_parse_immediate_values(self):
        trace = [
            "movq $0x1234, %rax",
            "addq $100, %rbx",
            "cmpq $-50, %rcx",
            "movq $0, %rdx"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 4)
        for insn in result:
            self.assertTrue(len(insn.operands) == 2)

    def test_parse_register_operands(self):
        trace = [
            "movb %al, %bl",
            "movw %ax, %bx",
            "movl %eax, %ebx",
            "movq %rax, %rbx",
            "movq %r8, %r9",
            "movl %r10d, %r11d"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), len(trace))
        for i, insn in enumerate(result):
            self.assertEqual(insn.mnemonic, trace[i].split()[0])
            self.assertEqual(len(insn.operands), 2)

    def test_parse_complex_addressing(self):
        trace = [
            "mov (%rax), %rbx",         
            "mov 8(%rax), %rbx",        
            "mov (%rax,%rcx), %rbx",    
            "mov (%rax,%rcx,2), %rbx",  
            "mov 8(%rax,%rcx,4), %rbx"  
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), len(trace))
        for insn in result:
            self.assertEqual(insn.mnemonic, 'movq')
            self.assertEqual(len(insn.operands), 2)

    def test_parse_floating_point_instructions(self):
        trace = [
            "addss %xmm0, %xmm1",
            "subsd %xmm2, %xmm3",
            "mulps %xmm4, %xmm5",
            "divpd %xmm6, %xmm7"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = ['addss', 'subsd', 'mulps', 'divpd']
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_vector_instructions(self):
        trace = [
            "movdqa %xmm0, %xmm1",
            "paddb %xmm2, %xmm3",
            "psubb %xmm4, %xmm5",
            "pmullw %xmm6, %xmm7"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = ['movdqa', 'paddb', 'psubb', 'pmullw']
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_operand_types(self):
        trace = ["mov %rax, %rbx"]
        parser = TraceParser(trace)
        result = parser.parse()
        
        insn = result[0]
        self.assertEqual(len(insn.operands), 2)

        op1, op2 = insn.operands
        self.assertEqual(op1.type, CS_OP_REG)
        self.assertEqual(op2.type, CS_OP_REG)

    def test_parse_immediate_operand(self):
        trace = ["mov $100, %rax"]
        parser = TraceParser(trace)
        result = parser.parse()
        
        insn = result[0]
        self.assertEqual(len(insn.operands), 2)

        op1, op2 = insn.operands
        self.assertEqual(CS_OP_IMM, op1.type)
        self.assertEqual(CS_OP_REG, op2.type)

    def test_parse_memory_operand(self):
        trace = ["mov (%rax), %rbx"]
        parser = TraceParser(trace)
        result = parser.parse()
        
        insn = result[0]
        self.assertEqual(len(insn.operands), 2)

        op1, op2 = insn.operands
        self.assertEqual(op1.type, CS_OP_MEM)
        self.assertEqual(op2.type, CS_OP_REG)

    def test_parse_empty_trace(self):
        parser = TraceParser([])
        result = parser.parse()
        self.assertEqual(result, [])

    def test_parse_whitespace_handling(self):
        trace = [
            "  mov   %rax,   %rbx  ",
            "\tadd\t%rcx,\t%rdx\t",
            "sub %rsi, %rdi"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 3)
        mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(mnemonics, ['movq', 'addq', 'subq'])

    def test_parse_prefix_instructions(self):
        trace = [
            "rep movsb",
            "lock add %rax, (%rbx)",
            "repne scasb",
            "movq %rax, %rbx"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), len(trace))

        def has_prefix(insn, prefix):
            return any(pre == prefix for pre in insn.prefix)

        self.assertTrue(has_prefix(result[0], X86_PREFIX_REP))
        self.assertTrue(has_prefix(result[1], X86_PREFIX_LOCK))
        self.assertTrue(has_prefix(result[2], X86_PREFIX_REPNE))

    def test_parse_segment_overrides(self):
        trace = [
            "movq %fs:(%rax), %rbx",
            "movq %gs:8(%rcx), %rdx"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 2)
        for insn in result:
            self.assertEqual(insn.mnemonic, 'movq')

    def test_parse_large_trace(self):
        base_instructions = [
            "movq %rax, %rbx",
            "addq $1, %rcx",
            "subq %rdx, %rsi",
            "mulq %rdi"
        ]
        large_trace = base_instructions * 250  # 1000 instructions

        parser = TraceParser(large_trace)
        result = parser.parse()
        
        self.assertEqual(len(result), 1000)
        for i in range(0, len(result), len(base_instructions)):
            if i + 3 < len(result):
                self.assertEqual(result[i].mnemonic, 'movq')
                self.assertEqual(result[i + 1].mnemonic, 'addq')
                self.assertEqual(result[i + 2].mnemonic, 'subq')
                self.assertEqual(result[i + 3].mnemonic, 'mulq')

    def test_parse_conditional_jumps(self):
        trace = [
            "je .+2",
            "jne .+2", 
            "jb .+2",
            "jae .+2",
            "jo .+2",
            "jno .+2"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        self.assertEqual(len(result), len(trace))
        expected_mnemonics = ['je', 'jne', 'jb', 'jae', 'jo', 'jno']
        self.assertEqual([insn.mnemonic for insn in result], expected_mnemonics)

    def test_parse_compare_and_test(self):
        trace = [
            "cmpq %rax, %rbx",
            "testq %rcx, %rdx",
            "cmpq $100, %rsi",
            "testq $0xff, %rdi"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = [line.split()[0] for line in trace]
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_string_instructions(self):
        trace = [
            "movsb",
            "movsw", 
            "movsd",
            "lodsb",
            "stosb",
            "scasb",
            "cmpsb"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = ['movsb', 'movsw', 'movsl', 'lodsb', 'stosb', 'scasb', 'cmpsb']
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_bit_manipulation(self):
        trace = [
            "btq $5, %rax",
            "btsq $3, %rbx", 
            "btrq $7, %rcx",
            "btcq $1, %rdx",
            "bsfq %rsi, %rdi",
            "bsrq %r8, %r9"
        ]
        parser = TraceParser(trace)
        result = parser.parse()

        expected_mnemonics = ['btq', 'btsq', 'btrq', 'btcq', 'bsfq', 'bsrq']
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_stack_operations(self):
        trace = [
            "pushq %rax",
            "popq %rbx",
            "pushfq",
            "popfq",
            "enter $16, $0",
            "leave"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = [line.split()[0] for line in trace]
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)

    def test_parse_system_instructions(self):
        trace = [
            "nop",
            "hlt", 
            "cli",
            "sti",
            "cld",
            "std"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        
        expected_mnemonics = ['nop', 'hlt', 'cli', 'sti', 'cld', 'std']
        actual_mnemonics = [insn.mnemonic for insn in result]
        self.assertEqual(actual_mnemonics, expected_mnemonics)


@unittest.skipUnless(KEYSTONE_AVAILABLE, "Keystone/Capstone not available")
class TestTraceParserErrorHandling(unittest.TestCase):

    def test_invalid_instruction_syntax(self):
        trace = ["invalid_instruction_xyz"]
        parser = TraceParser(trace)
        
        with self.assertRaises(Exception):
            parser.parse()

    def test_malformed_operands(self):
        trace = ["mov %invalid_reg, %rbx"]
        parser = TraceParser(trace)
        
        with self.assertRaises(Exception):
            parser.parse()

    def test_mixed_valid_invalid(self):
        trace = [
            "mov %rax, %rbx",      
            "invalid_instruction",
            "add %rcx, %rdx"     
        ]
        parser = TraceParser(trace)
        
        with self.assertRaises(Exception):
            parser.parse()


@unittest.skipUnless(KEYSTONE_AVAILABLE, "Keystone/Capstone not available")
class TestTraceParserIntegration(unittest.TestCase):

    def test_real_world_trace_example(self):
        trace = [
            "pushq %rbp",
            "movq %rsp, %rbp",
            "subq $16, %rsp",
            "movq $100, -8(%rbp)",
            "movq $200, -16(%rbp)",
            "movq -8(%rbp), %rax",
            "addq -16(%rbp), %rax",
            "movq %rax, -4(%rbp)",
            "movq -4(%rbp), %rax",
            "leave",
            "ret"
        ]
        parser = TraceParser(trace)

        result = parser.parse()
        self.assertEqual(len(result), 11)

        self.assertEqual(result[0].mnemonic, 'pushq')
        self.assertEqual(result[-1].mnemonic, 'retq')

        for i in range(1, len(result)):
            self.assertGreater(result[i].address, result[i-1].address)

    def test_function_call_sequence(self):
        trace = [
            "pushq %rdi",
            "pushq %rsi", 
            "movq %rsp, %rdi",
            "movabsq $0x100, %rsi",
            "callq *%rax",
            "popq %rsi",
            "popq %rdi"
        ]
        parser = TraceParser(trace)
        result = parser.parse()
        self.assertEqual(len(result), 7)
        mnemonics = [insn.mnemonic for insn in result]
        expected_mnemonic = ['pushq', 'pushq', 'movq', 'movabsq', 'callq', 'popq', 'popq']
        self.assertEqual(mnemonics, expected_mnemonic)

    def test_loop_structure(self):
        trace = [
            "movabsq $10, %rcx",
            "movabsq $0, %rax",
            "addq %rcx, %rax",
            "decq %rcx",
            "jne .+2",
            "movq %rax, %rbx"
        ]

        parser = TraceParser(trace)
        result = parser.parse()
        self.assertEqual(len(result), 6)
        mnemonics = [insn.mnemonic for insn in result]
        expected_mnemonics = ['movabsq', 'movabsq', 'addq', 'decq', 'jne', 'movq']
        self.assertEqual(mnemonics, expected_mnemonics)


if __name__ == '__main__':
    unittest.main()
