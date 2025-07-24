from keystone import *
from capstone import *
from capstone.x86_const import *

from dataclasses import dataclass


class OperandType:
    REG = "reg"
    IMM = "imm"
    MEM = "mem"


@dataclass
class Operand:
    type: OperandType
    reg: str = None
    imm: int = None
    mem: dict = None  # {'base': str, 'index': str, 'scale': int, 'disp': int}


@dataclass
class Instruction:
    address: int
    mnemonic: str
    operands: list
    size: int
    prefix: list


class TraceParser:
    def __init__(self, trace):
        # remove "endbr64" and "nop" instructions from the trace as they're not supported by Keystone
        self.trace = [
            inst for inst in trace if inst.strip() and inst not in ["endbr64", "nop"]
        ]

        self.ks = Ks(KS_ARCH_X86, KS_MODE_64)
        self.ks.syntax = KS_OPT_SYNTAX_ATT

        self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.md.detail = True
        self.md.syntax = CS_OPT_SYNTAX_ATT

    def _get_ops(self, insn):
        operators = []
        for op in insn.operands:
            operand = self._parse_operand(op, insn)
            if operand:
                operators.append(operand)
        return operators

    def _parse_operand(self, op, insn):
        if op.type == CS_OP_REG:
            return Operand(type=OperandType.REG, reg=insn.reg_name(op.reg))
        elif op.type == CS_OP_IMM:
            return Operand(type=OperandType.IMM, imm=op.imm)
        elif op.type == CS_OP_MEM:
            mem = {
                "base": insn.reg_name(op.mem.base) if op.mem.base != 0 else None,
                "index": insn.reg_name(op.mem.index) if op.mem.index != 0 else None,
                "scale": op.mem.scale if op.mem.scale != 1 else None,
                "disp": op.mem.disp,
            }
        return Operand(type=OperandType.MEM, mem=mem)

    def parse(self):
        if not self.trace:
            return []
        instructions = ";".join(self.trace)
        encoding, _ = self.ks.asm(instructions)
        disassembled = self.md.disasm(bytes(encoding), 0x1000)
        parsed_instructions = []
        for insn in disassembled:
            parsed_instructions.append(
                Instruction(
                    address=insn.address,
                    mnemonic=insn.mnemonic,
                    operands=self._get_ops(insn),
                    size=insn.size,
                    prefix=[pre for pre in insn.prefix if pre != 0],
                )
            )
        return parsed_instructions
