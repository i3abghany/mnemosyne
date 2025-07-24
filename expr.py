from dataclasses import dataclass
import logging


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


def optimize_expr(expr: Expr, state) -> Expr:
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
                elif e.op == '<<':
                    return Const(left.value << right.value)
                elif e.op == '>>':
                    return Const(left.value >> right.value)
                elif e.op == "&":
                    return Const(left.value & right.value)
                elif e.op == "|":
                    return Const(left.value | right.value)
                elif e.op == "^":
                    return Const(left.value ^ right.value)

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


def expand_expr(expr: Expr, state, visited: set = None) -> Expr:
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


def print_expr_tree(logger: logging.Logger, expr: Expr, indent: str = ""):
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
