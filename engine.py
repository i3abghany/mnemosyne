from dataclasses import dataclass
from typing import Union, List
import logging
import re

# Set up logging
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
        return self.mem.get(str(addr_expr), Mem(addr_expr))


def parse_address(addr_str: str, state: SymbolicState):
    # Pattern: optional_displacement(base_reg, index_reg, scale)
    # Updated to handle whitespace better and validate registers
    pattern = r"(?P<disp>[-+]?(?:0x[0-9a-f]+|\d+))?\s*\(\s*(?P<base>%[a-z0-9]+)?\s*(?:,\s*(?P<index>%[a-z0-9]+))?\s*(?:,\s*(?P<scale>[1248]))?\s*\)"

    m = re.match(pattern, addr_str.strip())
    if not m:
        raise ValueError(f"Invalid address mode: {addr_str}")

    components = []

    if m.group("base"):
        base_reg = m.group("base").strip('%')
        # Check if register is valid
        valid_regs = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi",
                      "rsp", "rbp"] + [f"r{i}" for i in range(8, 16)]
        if base_reg not in valid_regs:
            raise ValueError(f"Invalid register: {base_reg}")
        base_expr = state.read_reg(base_reg)
        components.append(base_expr)

    if m.group("index"):
        index_reg = m.group("index").strip('%')
        valid_regs = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi",
                      "rsp", "rbp"] + [f"r{i}" for i in range(8, 16)]
        if index_reg not in valid_regs:
            raise ValueError(f"Invalid register: {index_reg}")
        index_expr = state.read_reg(index_reg)
        if m.group("scale"):
            index_expr = BinOp("*", index_expr, Const(int(m.group("scale"))))
        components.append(index_expr)

    if m.group("disp"):
        disp = m.group("disp")
        disp_val = int(disp, 16) if disp.startswith(
            ("0x", "-0x")) else int(disp)
        components.append(Const(disp_val))

    if not components:
        raise ValueError("Empty address expression")

    addr = components[0]
    for component in components[1:]:
        addr = BinOp("+", addr, component)

    # Always ensure result is a BinOp for consistency
    if isinstance(addr, Var):
        addr = BinOp("+", addr, Const(0))
    elif isinstance(addr, Const):
        # For displacement-only addressing
        addr = BinOp("+", addr, Const(0))

    # Optimize the address expression. This ensures equivalent address
    # expressions are coalesced.
    addr = optimize_expr(addr, state)

    return addr


class SymbolicEngine:
    def __init__(self, state: SymbolicState = None):
        self.state = state or SymbolicState()

    def parse_trace_and_execute(self, trace: List[str]):
        """Parse and execute a trace of assembly instructions."""
        for line in trace:
            self._execute_instruction(line)

    def _execute_instruction(self, line: str):
        """Execute a single instruction line."""
        line = line.strip()
        if not line or line.startswith("endbr64") or line.startswith("ret") or line.startswith("nop") or line.startswith("cltq"):
            return

        if re.match(r"(cmp|test|jmp|j[a-z]+)", line):
            logger.debug(f"skipping: {line}")
            return

        line = re.sub(r"%e([a-z0-9]+)", r"%r\1", line)

        # xor[lq] %reg, %reg
        if m := re.match(r"xor[lq]? %([a-z0-9]+), %([a-z0-9]+)", line):
            src_reg, dst_reg = m[1], m[2]
            src_val = self.state.read_reg(src_reg)
            dst_val = self.state.read_reg(dst_reg)
            new_val = BinOp("^", dst_val, src_val)
            var, _ = self.state.write_reg(dst_reg, new_val)
            logger.info(f"{var} = {new_val}")
            return

        # lea[lq] $imm, %reg
        if m := re.match(r"lea[lq]? \$(-?(?:0x[0-9a-f]+|[0-9]+)), %([a-z0-9]+)", line):
            imm_str, reg = m[1], m[2]
            imm = int(imm_str, 16) if imm_str.startswith(
                ('-0x', '0x')) else int(imm_str)
            var, expr = self.state.write_reg(reg, Const(imm))
            logger.info(f"{var} = {expr}")
            return

        # lea[lq] mem, %reg
        if m := re.match(r"lea[lq]? (.+), %([a-z0-9]+)", line):
            addr_str, reg = m[1], m[2]
            addr = parse_address(addr_str, self.state)
            var, expr = self.state.write_reg(reg, addr)
            logger.info(f"{var} = {expr}")
            return

        # mov $imm, %reg
        if m := re.match(r"mov(?:abs)?[lq]? \$(-?(?:0x[0-9a-f]+|[0-9]+)), %([a-z0-9]+)", line):
            imm_str, reg = m[1], m[2]
            imm = int(imm_str, 16) if imm_str.startswith(
                ('-0x', '0x')) else int(imm_str)
            var, expr = self.state.write_reg(reg, Const(imm))
            logger.info(f"{var} = {expr}")
            return

        # mov %reg, %reg
        if m := re.match(r"mov[lq]? %([a-z0-9]+), %([a-z0-9]+)", line):
            src_reg, dst_reg = m[1], m[2]
            src_val = self.state.read_reg(src_reg)
            var, expr = self.state.write_reg(dst_reg, src_val)
            logger.info(f"{var} = {expr}")
            return

        # mov $imm, mem
        if m := re.match(r"mov[lq]? \$(-?(?:0x[0-9a-f]+|[0-9]+)), (.+)", line):
            imm_str, addr_str = m[1], m[2]
            imm = int(imm_str, 16) if imm_str.startswith(
                ('-0x', '0x')) else int(imm_str)
            addr = parse_address(addr_str, self.state)
            mem_var, _ = self.state.mem_store(addr, Const(imm))
            logger.info(f"{mem_var} = {Const(imm)}")
            return

        # mov %reg, mem
        if m := re.match(r"mov[lq]? %([a-z0-9]+), (.+)", line):
            reg, addr_str = m[1], m[2]
            reg_val = self.state.read_reg(reg)
            addr = parse_address(addr_str, self.state)
            mem_var, _ = self.state.mem_store(addr, reg_val)
            logger.info(f"{mem_var} = {reg_val}")
            return

        # mov mem, %reg
        if m := re.match(r"mov[lq]? (.+), %([a-z0-9]+)", line):
            addr_str, reg = m[1], m[2]
            addr = parse_address(addr_str, self.state)
            val = self.state.mem_load(addr)
            var, _ = self.state.write_reg(reg, val)
            logger.info(f"{var} = {val}")
            return

        # add $imm, %reg
        if m := re.match(r"add[lq]? \$(-?(?:0x[0-9a-f]+|[0-9]+)), %([a-z0-9]+)", line):
            imm_str, reg = m[1], m[2]
            imm = int(imm_str, 16) if imm_str.startswith(
                ('-0x', '0x')) else int(imm_str)
            reg_val = self.state.read_reg(reg)
            new_val = BinOp("+", reg_val, Const(imm))
            var, _ = self.state.write_reg(reg, new_val)
            logger.info(f"{var} = {new_val}")
            return

        # add %reg1, %reg2
        if m := re.match(r"add[lq]? %([a-z0-9]+), %([a-z0-9]+)", line):
            src_reg, dst_reg = m[1], m[2]
            src_val = self.state.read_reg(src_reg)
            dst_val = self.state.read_reg(dst_reg)
            new_val = BinOp("+", dst_val, src_val)
            var, _ = self.state.write_reg(dst_reg, new_val)
            logger.info(f"{var} = {new_val}")
            return

        # add $imm, mem
        if m := re.match(r"add[lq]? \$(-?(?:0x[0-9a-f]+|[0-9]+)), (.+)", line):
            imm_str, addr_str = m[1], m[2]
            imm = int(imm_str, 16) if imm_str.startswith(
                ('-0x', '0x')) else int(imm_str)
            addr = parse_address(addr_str, self.state)
            old_val = self.state.mem_load(addr)
            new_val = BinOp("+", old_val, Const(imm))
            mem_var, _ = self.state.mem_store(addr, new_val)
            logger.info(f"{mem_var} = {new_val}")
            return

        # add %reg, mem
        if m := re.match(r"add[lq]? %([a-z0-9]+), (.+)", line):
            reg, addr_str = m[1], m[2]
            reg_val = self.state.read_reg(reg)
            addr = parse_address(addr_str, self.state)
            old_val = self.state.mem_load(addr)
            new_val = BinOp("+", old_val, reg_val)
            mem_var, _ = self.state.mem_store(addr, new_val)
            logger.info(f"{mem_var} = {new_val}")
            return

        # add mem, %reg
        if m := re.match(r"add[lq]? (.+), %([a-z0-9]+)", line):
            addr_str, reg = m[1], m[2]
            addr = parse_address(addr_str, self.state)
            mem_val = self.state.mem_load(addr)
            reg_val = self.state.read_reg(reg)
            new_val = BinOp("+", reg_val, mem_val)
            var, _ = self.state.write_reg(reg, new_val)
            logger.info(f"{var} = {new_val}")
            return

        logger.warning(f"unhandled line: {line}")


def expand_expr(expr: Expr, state: SymbolicState) -> Expr:
    if isinstance(expr, Const):
        return expr

    if isinstance(expr, Var):
        key = f"{expr.name}_{expr.version}"
        if key in state.definitions:
            return expand_expr(state.definitions[key], state)
        else:
            return expr

    if isinstance(expr, BinOp):
        left = expand_expr(expr.left, state)
        right = expand_expr(expr.right, state)
        return BinOp(expr.op, left, right)

    if isinstance(expr, Mem):
        addr = expand_expr(expr.addr, state)
        val = state.mem.get(str(addr))
        if val:
            return expand_expr(val, state)
        else:
            return Mem(addr)

    return expr


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
    trace = [line.split(':')[1]
             for line in trace if ':' in line and not line.startswith('#')]
    return trace


if __name__ == "__main__":
    trace = get_trace_from_file('a.log')
    logger.debug(f"Trace: {trace}")

    engine = SymbolicEngine()
    engine.parse_trace_and_execute(trace)

    # Access results from the engine's state
    expr = engine.state.current_var('rax')
    expanded = expand_expr(expr, engine.state)
    optimized = optimize_expr(expanded, engine.state)
    print(f"Final rax value: {optimized}")
