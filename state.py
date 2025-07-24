from expr import Expr, Var, Const, BinOp, Mem


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
