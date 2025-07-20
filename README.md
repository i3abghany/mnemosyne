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

## Trace Generation from Source Code

Mnemosyne supports generating dynamic traces from C source code using QEMU and a custom plugin.

### Prerequisites

1. **QEMU**: Mnemosyne is tested with QEMU version 10.0.2
2. **Dyntrace Plugin**: A custom QEMU plugin for dynamic tracing is required. This plugin is included in the Mnemosyne repository.
3. **QEMU Build Tools**: Ensure you have the necessary build tools installed (e.g., `gcc`, `ninja`, `git`).
4. **Capstone**: QEMU must be built with Capstone support for disassembly.

### Setup

```bash
# Install dependencies
$ sudo apt install build-essential git ninja-build libcapstone-dev

# Clone QEMU and apply the dyntrace plugin patch
$ git clone git@github.com:qemu/qemu.git /path/to/qemu
$ cd /path/to/qemu
$ git reset --hard v10.0.2
$ git am /path/to/mnemosyne/qemu-dyntrace-plugin.patch

# Build QEMU with the dyntrace plugin
$ mkdir -p build && cd build
$ ../configure --enable-plugins --enable-capstone --target-list=x86_64-linux-user
$ ninja
```

### Generating Traces

To generate traces from C source code, use the `gen.sh` script provided in the Mnemosyne repository. This script will compile the C source code with QEMU and produce a trace file. Currently, it only keeps the assembly instructions of the `main` function.

```bash
$ chmod +x gen.sh
$ QEMU_PATH=/path/to/qemu ./gen.sh example.c
```

This produces a filtered trace file (`example.log`) containing only the main function's assembly instructions. Use `get_trace_from_file` in `engine.py` to load this trace, and then execute it with the `SymbolicEngine` similar to the example above.

## Requirements

- Mnemosyne is tested on Python 3.13.2
- optionally: `coverage` for test coverage reports
