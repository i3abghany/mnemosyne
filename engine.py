from dataclasses import dataclass
import sys
from typing import Union, List
import logging
import re
import os

from keystone import *
from capstone import *
from capstone.x86_const import *

from parser import TraceParser, OperandType
from expr import Expr, Const, Var, Mem, BinOp, optimize_expr, expand_expr
from state import SymbolicState

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def normalize_register_name(reg_name: str) -> str:
    reg_map = {
        'eax': 'rax', 'ebx': 'rbx', 'ecx': 'rcx', 'edx': 'rdx',
        'esi': 'rsi', 'edi': 'rdi', 'esp': 'rsp', 'ebp': 'rbp',
        'r8d': 'r8', 'r9d': 'r9', 'r10d': 'r10', 'r11d': 'r11',
        'r12d': 'r12', 'r13d': 'r13', 'r14d': 'r14', 'r15d': 'r15'
    }
    return reg_map.get(reg_name, reg_name)


def build_memory_address(mem_operand, state: SymbolicState) -> Expr:
    components = []

    # Base register
    if mem_operand['base']:
        base_name = normalize_register_name(mem_operand['base'])
        base_expr = state.read_reg(base_name)
        components.append(base_expr)

    # Displacement
    if mem_operand['disp'] != 0:
        components.append(Const(mem_operand['disp']))

    # Index register with scale
    if mem_operand['index']:
        index_name = normalize_register_name(mem_operand['index'])
        index_expr = state.read_reg(index_name)
        if mem_operand['scale'] is not None and mem_operand['scale'] > 1:
            index_expr = BinOp("*", index_expr, Const(mem_operand['scale']))
        components.append(index_expr)

    if not components:
        return Const(0)

    addr = components[0]
    for component in components[1:]:
        addr = BinOp("+", addr, component)

    return addr


def operand_to_expr(operand, state: SymbolicState) -> Expr:
    if operand.type == OperandType.REG:
        reg_name = operand.reg
        reg_name = normalize_register_name(reg_name)
        return state.read_reg(reg_name)

    elif operand.type == OperandType.IMM:
        return Const(operand.imm)

    elif operand.type == OperandType.MEM:
        addr = build_memory_address(operand.mem, state)
        if isinstance(addr, Mem):
            return addr
        return Mem(addr)

    else:
        raise ValueError(f"Unsupported operand type: {operand.type}")


def operand_to_lvalue(operand, state: SymbolicState) -> tuple:
    if operand.type == OperandType.REG:
        reg_name = operand.reg
        reg_name = normalize_register_name(reg_name)
        return ('reg', reg_name)

    elif operand.type == OperandType.MEM:
        addr_expr = operand_to_expr(operand, state)
        return ('mem', addr_expr)

    else:
        raise ValueError(f"Cannot write to operand type: {operand.type}")


class SymbolicEngine:
    def __init__(self, state: SymbolicState = None):
        self.state = state or SymbolicState()
        self.parser = None

    def parse_trace_and_execute(self, trace: List[str]):
        parsed_instructions = self.parser = TraceParser(trace).parse()
        for instruction in parsed_instructions:
            self._execute_instruction(instruction)

    def _execute_instruction(self, instruction):
        mnemonic = instruction.mnemonic
        operands = instruction.operands

        if mnemonic in ['retq', 'cltq']:
            logger.debug(f"Skipping: {mnemonic}")
            return

        if mnemonic.startswith(('cmp', 'test', 'j')):
            logger.debug(f"Skipping: {mnemonic}")
            return

        try:
            # MOV instructions
            if mnemonic.startswith('mov'):
                self._handle_mov(operands)

            # LEA instructions
            elif mnemonic.startswith('lea'):
                self._handle_lea(operands)

            # Arithmetic instructions
            elif mnemonic.startswith(('add', 'sub', 'xor', 'and', 'or', 'imul', 'shl', 'shr', 'sar')):
                self._handle_arithmetic(instruction, operands)

            else:
                logger.warning(f"Unhandled instruction: {mnemonic}")

        except Exception as e:
            logger.error(f"Error executing {mnemonic}: {e}")

    def _handle_mov(self, operands):
        if len(operands) != 2:
            logger.warning(f"MOV with {len(operands)} operands")
            return

        src_operand, dst_operand = operands

        # Handle source operand - if it's memory, we need to load from it
        if src_operand.type == OperandType.MEM:
            addr = build_memory_address(src_operand.mem, self.state)
            addr = optimize_expr(addr, self.state)
            src_expr = self.state.mem_load(addr)
        else:
            src_expr = operand_to_expr(src_operand, self.state)

        # Get destination
        dst_type, dst_id = operand_to_lvalue(
            dst_operand, self.state)

        if dst_type == 'reg':
            var, _ = self.state.write_reg(dst_id, src_expr)
            logger.info(f"{var} = {src_expr}")
        elif dst_type == 'mem':
            # dst_id is a Mem(addr) object, we need just the address
            if isinstance(dst_id, Mem):
                addr = optimize_expr(dst_id.addr, self.state)
            else:
                addr = optimize_expr(dst_id, self.state)
            mem_var, _ = self.state.mem_store(addr, src_expr)
            logger.info(f"{mem_var} = {src_expr}")

    def _handle_lea(self, operands):
        if len(operands) != 2:
            logger.warning(f"LEA with {len(operands)} operands")
            return

        src_operand, dst_operand = operands

        # For LEA, we want the address itself, not the memory content
        if src_operand.type == OperandType.MEM:
            addr_expr = build_memory_address(src_operand.mem, self.state)
        elif src_operand.type == OperandType.IMM:
            addr_expr = Const(src_operand.imm)
        else:
            logger.warning(
                f"Unexpected LEA source operand type: {src_operand.type}")
            return

        # Get destination
        dst_type, dst_id = operand_to_lvalue(dst_operand, self.state)

        if dst_type == 'reg':
            addr_expr = optimize_expr(addr_expr, self.state)
            var, _ = self.state.write_reg(dst_id, addr_expr)
            logger.info(f"{var} = {addr_expr}")
        else:
            logger.warning("LEA to memory not supported")

    def _handle_arithmetic(self, instruction, operands):
        if len(operands) != 2:
            logger.warning(
                f"{instruction.mnemonic} with {len(operands)} operands")
            return

        src_operand, dst_operand = operands
        mnemonic = instruction.mnemonic

        # Map mnemonics to operators
        op_map = {
            'addq': '+', 'addl': '+',
            'subq': '-', 'subl': '-',
            'xorq': '^', 'xorl': '^',
            'andq': '&', 'andl': '&',
            'orq': '|', 'orl': '|',
            'shlq': '<<', 'shll': '<<',
            'shrq': '>>', 'shrl': '>>',
            'sarq': '>>', 'sarl': '>>',
            'imulq': '*', 'imull': '*'
        }

        if mnemonic not in op_map:
            logger.warning(f"Unknown arithmetic operation: {mnemonic}")
            return

        sym_op = op_map[mnemonic]

        # Get source value
        src_expr = operand_to_expr(src_operand, self.state)

        # Get destination
        dst_type, dst_id = operand_to_lvalue(dst_operand, self.state)

        if dst_type == 'reg':
            # Read current value of destination register
            dst_expr = operand_to_expr(dst_operand, self.state)
            new_expr = BinOp(sym_op, dst_expr, src_expr)
            var, _ = self.state.write_reg(dst_id, new_expr)
            logger.info(f"{var} = {new_expr}")

        elif dst_type == 'mem':
            # Read current value from memory
            # dst_id is a Mem(addr) object, we need just the address
            if isinstance(dst_id, Mem):
                addr = optimize_expr(dst_id.addr, self.state)
            else:
                addr = optimize_expr(dst_id, self.state)
            old_val = self.state.mem_load(addr)
            new_val = BinOp(sym_op, old_val, src_expr)
            mem_var, _ = self.state.mem_store(addr, new_val)
            logger.info(f"{mem_var} = {new_val}")
        else:
            logger.warning(f"Unsupported destination type: {dst_type}")


def get_trace_from_file(file_path):
    trace = open(file_path).readlines()
    trace = [line.strip() for line in trace if line.strip()]
    result = []
    for line in trace:
        if line.startswith('#'):
            continue
        if ':' in line:
            result.append(line.split(':', 1)[1].strip())
        else:
            result.append(line)
    return result


if __name__ == "__main__":
    trace = get_trace_from_file('a.log')
    logger.debug(f"Trace: {trace}")

    engine = SymbolicEngine()
    engine.parse_trace_and_execute(trace)

    expr = engine.state.current_var('rax')
    expanded = expand_expr(expr, engine.state)
    optimized = optimize_expr(expanded, engine.state)
    print(f'Expanded rax: {expanded}')
    print(f"Final rax value: {optimized}")
