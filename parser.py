from keystone import *
from capstone import *
from capstone.x86_const import *

from dataclasses import dataclass


@dataclass
class Instruction:
    address: int
    mnemonic: str
    operands: list
    size: int
    prefix: list


class TraceParser:
    def __init__(self, trace):
        self.trace = trace

        self.ks = Ks(KS_ARCH_X86, KS_MODE_64)
        self.ks.syntax = KS_OPT_SYNTAX_ATT

        self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.md.detail = True
        self.md.syntax = CS_OPT_SYNTAX_ATT

    def parse(self):
        if not self.trace:
            return []
        instructions = ';'.join(self.trace)
        encoding, _ = self.ks.asm(instructions)
        disassembled = self.md.disasm(bytes(encoding), 0x1000)
        parsed_instructions = []
        for insn in disassembled:
            parsed_instructions.append(Instruction(
                address=insn.address,
                mnemonic=insn.mnemonic,
                operands=[op for op in insn.operands],
                size=insn.size,
                prefix=[pre for pre in insn.prefix if pre != 0],
            ))
        return parsed_instructions
