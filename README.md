# Mnemosyne

A trace-based symbolic execution engine for x86-64 assembly.

## Usage

### Using the SymbolicEngine Class

```python
from engine import SymbolicEngine, expand_expr, optimize_expr

# Create and use the engine
engine = SymbolicEngine()

# Execute assembly trace
trace = [
    "mov $42, %rax",
    "add $8, %rax", 
    "mov %rax, (%rsp)"
]

engine.parse_trace_and_execute(trace)

rax_var = engine.state.current_var('rax')
final_value = optimize_expr(expand_expr(rax_var, engine.state), engine.state)
print(f"rax: {final_value}")
```

### Using Custom Initial State

```python
from engine import SymbolicEngine, SymbolicState, Const

# Create state with initial values
state = SymbolicState()
state.write_reg('rcx', Const(100))

# Create engine with custom state
engine = SymbolicEngine(state)
engine.parse_trace_and_execute(["mov %rcx, %rax"])
```

## Testing

```bash
# Run all tests
python run_tests.py

# Run quick tests
python run_tests.py --quick
```

## Requirements

- Mnemosyne is tested on Python 3.13.2
- optionally: `coverage` for test coverage reports
