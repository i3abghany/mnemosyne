from dataclasses import dataclass
import sys
from typing import Union, List
import logging
import re
import os

from keystone import *
from capstone import *
from capstone.x86_const import *

from parser import TraceParser, OperandType, Operand, Instruction

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Expr:
    def __str__(self):
        raise NotImplementedError


@dataclass
class Const(Expr):
    value: int

    def __str__(self):
        return str(self.value)


@dataclass
class Var(Expr):
    name: str
    version: int

    def __str__(self):
        return f"{self.name}_{self.version}"


@dataclass
class Mem(Expr):
    addr: Expr

    def __str__(self):
        return f"mem[{self.addr}]"


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"


class SymbolicState:
    def __init__(self):
        self.reg_versions = {}
        self.mem = {}
        self.definitions = {}
        self.rsp_val = Const(0x1000)
        self.definitions["rsp_1"] = self.rsp_val

    def new_var(self, name):
        version = self.reg_versions.get(name, 0) + 1
        self.reg_versions[name] = version
        return Var(name, version)

    def current_var(self, name):
        version = self.reg_versions.get(name, 0)
        return Var(name, version)

    def write_reg(self, name, expr: Expr):
        var = self.new_var(name)
        key = f"{var.name}_{var.version}"
        self.definitions[key] = expr
        return var, expr

    def read_reg(self, name):
        return self.current_var(name)

    def mem_store(self, addr_expr: Expr, value: Expr):
        key = str(addr_expr)
        self.mem[key] = value

        # Create symbolic memory variable
        name = f"mem[{key}]"
        version = self.reg_versions.get(name, 0) + 1
        self.reg_versions[name] = version
        mem_var = Var(name, version)

        # Store definition
        def_key = f"{mem_var.name}_{mem_var.version}"
        self.definitions[def_key] = value

        return mem_var, value

    def mem_load(self, addr_expr: Expr) -> Expr:
        key = str(addr_expr)
        if key in self.mem:
            # Return the latest version of this memory location
            name = f"mem[{key}]"
            version = self.reg_versions.get(name, 0)
            if version > 0:
                return Var(name, version)
        return Mem(addr_expr)


def normalize_register_name(reg_name: str) -> str:
    reg_map = {
        'eax': 'rax', 'ebx': 'rbx', 'ecx': 'rcx', 'edx': 'rdx',
        'esi': 'rsi', 'edi': 'rdi', 'esp': 'rsp', 'ebp': 'rbp',
        'r8d': 'r8', 'r9d': 'r9', 'r10d': 'r10', 'r11d': 'r11',
        'r12d': 'r12', 'r13d': 'r13', 'r14d': 'r14', 'r15d': 'r15'
    }
    return reg_map.get(reg_name, reg_name)


def operand_to_expr(operand, instruction, state: SymbolicState) -> Expr:
    if operand.type == OperandType.REG:
        reg_name = operand.reg
        reg_name = normalize_register_name(reg_name)
        return state.read_reg(reg_name)

    elif operand.type == OperandType.IMM:
        return Const(operand.imm)

    elif operand.type == OperandType.MEM:
        components = []

        # Base register
        if operand.mem['base']:
            base_name = operand.mem['base']
            base_name = normalize_register_name(base_name)
            base_expr = state.read_reg(base_name)
            components.append(base_expr)

        if operand.mem['disp'] != 0:
            components.append(Const(operand.mem['disp']))

        # Index register with scale
        if operand.mem['index']:
            index_name = operand.mem['index']
            index_name = normalize_register_name(index_name)
            index_expr = state.read_reg(index_name)
            if operand.mem['scale'] is not None and operand.mem['scale'] > 1:
                index_expr = BinOp(
                    "*", index_expr, Const(operand.mem['scale']))
            components.append(index_expr)

        if not components:
            return Const(0)

        addr = components[0]
        for component in components[1:]:
            addr = BinOp("+", addr, component)

        if isinstance(addr, Mem):
            return addr
        return Mem(addr)

    else:
        raise ValueError(f"Unsupported operand type: {operand.type}")


def operand_to_lvalue(operand, instruction, state: SymbolicState) -> tuple:
    if operand.type == OperandType.REG:
        reg_name = operand.reg
        reg_name = normalize_register_name(reg_name)
        return ('reg', reg_name)

    elif operand.type == OperandType.MEM:
        addr_expr = operand_to_expr(operand, instruction, state)
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
                self._handle_mov(instruction, operands)

            # LEA instructions
            elif mnemonic.startswith('lea'):
                self._handle_lea(instruction, operands)

            # Arithmetic instructions
            elif mnemonic.startswith(('add', 'sub', 'xor', 'and', 'or', 'imul', 'shl', 'shr', 'sar')):
                self._handle_arithmetic(instruction, operands)

            else:
                logger.warning(f"Unhandled instruction: {mnemonic}")

        except Exception as e:
            logger.error(f"Error executing {mnemonic}: {e}")

    def _handle_mov(self, instruction, operands):
        if len(operands) != 2:
            logger.warning(f"MOV with {len(operands)} operands")
            return

        src_operand, dst_operand = operands

        # Handle source operand - if it's memory, we need to load from it
        if src_operand.type == OperandType.MEM:
            # Get the address expression
            addr_components = []
            if src_operand.mem['base']:
                base_name = normalize_register_name(src_operand.mem['base'])
                base_expr = self.state.read_reg(base_name)
                addr_components.append(base_expr)

            if src_operand.mem['disp'] != 0:
                addr_components.append(Const(src_operand.mem['disp']))

            if src_operand.mem['index']:
                index_name = normalize_register_name(src_operand.mem['index'])
                index_expr = self.state.read_reg(index_name)
                if src_operand.mem['scale'] is not None:
                    if src_operand.mem['scale'] > 1:
                        index_expr = BinOp(
                            "*", index_expr, Const(src_operand.mem['scale']))
                addr_components.append(index_expr)

            if not addr_components:
                addr = Const(0)
            else:
                addr = addr_components[0]
                for component in addr_components[1:]:
                    addr = BinOp("+", addr, component)

            addr = optimize_expr(addr, self.state)
            src_expr = self.state.mem_load(addr)
        else:
            src_expr = operand_to_expr(
                src_operand, instruction, self.state)

        # Get destination
        dst_type, dst_id = operand_to_lvalue(
            dst_operand, instruction, self.state)

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

    def _handle_lea(self, instruction, operands):
        if len(operands) != 2:
            logger.warning(f"LEA with {len(operands)} operands")
            return

        src_operand, dst_operand = operands

        # For LEA, we want the address itself, not the memory content
        if src_operand.type == CS_OP_MEM:
            components = []

            # Base register
            if src_operand.mem.base != 0:
                base_name = instruction.reg_name(src_operand.mem.base)
                base_name = normalize_register_name(base_name)
                base_expr = self.state.read_reg(base_name)
                components.append(base_expr)

            # Index register with scale
            if src_operand.mem.index != 0:
                index_name = instruction.reg_name(src_operand.mem.index)
                index_name = normalize_register_name(index_name)
                index_expr = self.state.read_reg(index_name)
                if src_operand.mem.scale is not None and src_operand.mem.scale > 1:
                    index_expr = BinOp(
                        "*", index_expr, Const(src_operand.mem.scale))
                components.append(index_expr)

            # Displacement
            if src_operand.mem.disp != 0:
                components.append(Const(src_operand.mem.disp))

            if not components:
                addr_expr = Const(0)
            else:
                addr_expr = components[0]
                for component in components[1:]:
                    addr_expr = BinOp("+", addr_expr, component)

        elif src_operand.type == CS_OP_IMM:
            addr_expr = Const(src_operand.imm)
        else:
            logger.warning(
                f"Unexpected LEA source operand type: {src_operand.type}")
            return

        # Get destination
        dst_type, dst_id = operand_to_lvalue(
            dst_operand, instruction, self.state)

        if dst_type == 'reg':
            addr_expr = optimize_expr(addr_expr, self.state)
            var, _ = self.state.write_reg(dst_id, addr_expr)
            logger.info(f"{var} = {addr_expr}")
        else:
            logger.warning("LEA to memory not supported")

    def _handle_arithmetic(self, instruction, operands):
        """Handle arithmetic instructions using Capstone operands."""
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
        src_expr = operand_to_expr(
            src_operand, instruction, self.state)

        # Get destination
        dst_type, dst_id = operand_to_lvalue(
            dst_operand, instruction, self.state)

        if dst_type == 'reg':
            # Read current value of destination register
            dst_expr = operand_to_expr(
                dst_operand, instruction, self.state)
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


def expand_expr(expr: Expr, state: SymbolicState, visited: set = None) -> Expr:
    if visited is None:
        visited = set()

    expr_str = str(expr)
    if expr_str in visited:
        return expr

    visited.add(expr_str)

    try:
        if isinstance(expr, Const):
            return expr

        if isinstance(expr, Var):
            key = f"{expr.name}_{expr.version}"
            if key in state.definitions:
                return expand_expr(state.definitions[key], state, visited)
            else:
                return expr

        if isinstance(expr, BinOp):
            left = expand_expr(expr.left, state, visited)
            right = expand_expr(expr.right, state, visited)
            return BinOp(expr.op, left, right)

        if isinstance(expr, Mem):
            addr = expand_expr(expr.addr, state, visited)
            key = str(addr)
            if key in state.mem:
                stored_value = state.mem[key]
                stored_str = str(stored_value)
                mem_ref = f"mem[{key}]"
                if mem_ref in stored_str:
                    return Mem(addr)
                else:
                    return expand_expr(stored_value, state, visited)
            return Mem(addr)

        return expr
    finally:
        visited.remove(expr_str)


def optimize_expr(expr: Expr, state: SymbolicState) -> Expr:
    expanded = expand_expr(expr, state)

    def simplify(e: Expr) -> Expr:
        if isinstance(e, Const):
            return e

        elif isinstance(e, Var):
            return e

        elif isinstance(e, Mem):
            return Mem(simplify(e.addr))

        elif isinstance(e, BinOp):
            left = simplify(e.left)
            right = simplify(e.right)

            # Constant folding
            if isinstance(left, Const) and isinstance(right, Const):
                if e.op == "+":
                    return Const(left.value + right.value)
                elif e.op == "-":
                    return Const(left.value - right.value)
                elif e.op == "*":
                    return Const(left.value * right.value)
                elif e.op == "/":
                    return Const(left.value // right.value)

            if e.op == "+":
                if isinstance(left, Const) and left.value == 0:
                    return right
                if isinstance(right, Const) and right.value == 0:
                    return left
            if e.op == "-":
                if isinstance(right, Const) and right.value == 0:
                    return left
            if e.op == "*":
                if isinstance(left, Const) and left.value == 1:
                    return right
                if isinstance(right, Const) and right.value == 1:
                    return left
                if (isinstance(left, Const) and left.value == 0) or (isinstance(right, Const) and right.value == 0):
                    return Const(0)

            if e.op == "+":
                const_val = 0
                terms = []

                def collect_terms(x):
                    nonlocal const_val
                    if isinstance(x, Const):
                        const_val += x.value
                    elif isinstance(x, BinOp) and x.op == "+":
                        collect_terms(x.left)
                        collect_terms(x.right)
                    else:
                        terms.append(x)

                collect_terms(left)
                collect_terms(right)

                result = Const(const_val) if const_val != 0 else None
                for t in terms:
                    result = BinOp("+", result, t) if result else t

                return result if result else Const(0)

            return BinOp(e.op, left, right)

        return e

    return simplify(expanded)


def print_expr_tree(expr: Expr, indent: str = ""):
    logger.info(f"{indent}{expr}")

    if isinstance(expr, Var) or isinstance(expr, Const):
        return

    if isinstance(expr, BinOp):
        print_expr_tree(expr.left, indent + "  ")
        print_expr_tree(expr.right, indent + "  ")
    elif isinstance(expr, Mem):
        print_expr_tree(expr.addr, indent + "  ")
    else:
        logger.warning(f"Unknown expression type: {type(expr)}")


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
