#!/usr/bin/python3


import argparse
from collections import namedtuple
import re


# Some helpful constant values that we'll be using.
Constants = namedtuple("Constants",["NUM_REGS", "MEM_SIZE", "REG_SIZE"])
constants = Constants(NUM_REGS = 8,
                      MEM_SIZE = 2**13,
                      REG_SIZE = 2**16)

def load_machine_code(machine_code, mem):
    """
    Loads an E20 machine code file into the list
    provided by mem. We assume that mem is
    large enough to hold the values in the machine
    code file.
    sig: list(str) -> list(int) -> NoneType
    """
    machine_code_re = re.compile("^ram\[(\d+)\] = 16'b(\d+);.*$")
    expectedaddr = 0
    for line in machine_code:
        match = machine_code_re.match(line)
        if not match:
            raise ValueError("Can't parse line: %s" % line)
        addr, instr = match.groups()
        addr = int(addr,10)
        instr = int(instr,2)
        if addr != expectedaddr:
            raise ValueError("Memory addresses encountered out of sequence: %s" % addr)
        if addr >= len(mem):
            raise ValueError("Program too big for memory")
        expectedaddr += 1
        mem[addr] = instr


def print_cache_config(cache_name, size, assoc, blocksize, num_rows):
    """
    Prints out the correctly-formatted configuration of a cache.

    cache_name -- The name of the cache. "L1" or "L2"

    size -- The total size of the cache, measured in memory cells.
        Excludes metadata

    assoc -- The associativity of the cache. One of [1,2,4,8,16]

    blocksize -- The blocksize of the cache. One of [1,2,4,8,16,32,64])

    num_rows -- The number of rows in the given cache.

    sig: str, int, int, int, int -> NoneType
    """

    summary = "Cache %s has size %s, associativity %s, " \
        "blocksize %s, rows %s" % (cache_name,
        size, assoc, blocksize, num_rows)
    print(summary)

def print_log_entry(cache_name, status, pc, addr, row):
    """
    Prints out a correctly-formatted log entry.

    cache_name -- The name of the cache where the event
        occurred. "L1" or "L2"

    status -- The kind of cache event. "SW", "HIT", or
        "MISS"

    pc -- The program counter of the memory
        access instruction

    addr -- The memory address being accessed.

    row -- The cache row or set number where the data
        is stored.

    sig: str, str, int, int, int -> NoneType
    """
    log_entry = "{event:8s} pc:{pc:5d}\taddr:{addr:5d}\t" \
        "row:{row:4d}".format(row=row, pc=pc, addr=addr,
            event = cache_name + " " + status)
    print(log_entry)


#simulater function. According to the opcode,
#split instruction into three possible functions to analyze
def simulation(pc, regs, memory, cache):
    instruction = memory[pc % 8192]
    # Instructions with three register arguments
    if (instruction & 57344) >> 13 == 0b000:
        return instruction_3R(instruction, pc, regs, memory, cache)
    # Instructions with no register arguments
    elif ((instruction & 57344) >> 13) == 0b010 or ((instruction & 57344) >> 13 == 0b011):
        return instruction_0R(instruction, pc, regs, memory, cache)
    #Instructions with two register arguments
    else:
        return instruction_2R(instruction, pc, regs, memory, cache)


#if the given instruction with three register arguments
def instruction_3R(instruction, pc, regs, memory, cache):
    regSrcA = (instruction & 7168) >> 10
    regSrcB = (instruction & 896) >> 7
    regDst = (instruction & 112) >> 4
    opcode = (instruction & 15)

    if opcode == 0b1000: #jr
        pc = regs[regSrcA]
        return valid_pc(pc), regs, memory, cache

    # if a program tries to change the value of $0, do nothing
    if regDst == 0b0:
        return valid_pc(pc + 1), regs, memory, cache

    if opcode == 0b0000:
        regs[regDst] = keep_16bits(regs[regSrcA] + regs[regSrcB])
    elif opcode == 0b0001:
        regs[regDst] = keep_16bits(regs[regSrcA] - regs[regSrcB])
    elif opcode == 0b0010:
        regs[regDst] = keep_16bits(regs[regSrcA] | regs[regSrcB])
    elif opcode == 0b0011:
        regs[regDst] = keep_16bits(regs[regSrcA] & regs[regSrcB])
    elif opcode == 0b0100:
        if regs[regSrcA] < regs[regSrcB]:
            regs[regDst] = 0b1
        else:
            regs[regDst] = 0b0
    return valid_pc(pc + 1), regs, memory, cache


#if given instruction with two register arguments
def instruction_2R(instruction, pc, regs, memory, cache):
    opcode = (instruction & 57344) >> 13
    regSrc = (instruction & 7168) >> 10
    regDst = (instruction & 896) >> 7
    imm = instruction & 127

    # if a program tries to change the value of $0, do nothing
    if regDst == 0b000:
        if opcode == 0b111 or opcode == 0b100 or opcode == 0b001:
            return valid_pc(pc + 1), regs, memory, cache

    if opcode == 0b111: #slti
        if regs[regSrc] < sign_extend_7(imm):
            regs[regDst] = 0b1
        else:
            regs[regDst] = 0b0
        return valid_pc(pc + 1), regs, memory, cache
    elif opcode == 0b100: #lw
        val = (regs[regSrc] + sign_number_converter(imm, 7)) & 8191
        regs[regDst] = keep_16bits(memory[val])
        cache = load_data(cache, val, pc)
        return valid_pc(pc + 1), regs, memory, cache
    elif opcode == 0b101: #sw
        val = (regs[regSrc] + sign_number_converter(imm, 7)) & 8191
        memory[val] = regs[regDst]
        cache = store_data(cache, pc, val)
        return valid_pc(pc + 1), regs, memory, cache
    elif opcode == 0b110: # jeq
        if regs[regSrc] == regs[regDst]:
            pc = (pc + 1 + sign_number_converter(imm, 7))
            return valid_pc(pc), regs, memory, cache
        else:
            return valid_pc(pc + 1), regs, memory, cache
    elif opcode == 0b001: #addi
        regs[regDst] = keep_16bits(regs[regSrc] + sign_number_converter(imm, 7))
        return valid_pc(pc + 1), regs, memory, cache


#given instruction with no register arguments
def instruction_0R(instruction, pc, regs, memory, cache):
    opcode = (instruction & 57344) >> 13
    imm = instruction & 8191
    if opcode == 0b010:
        pc = imm
    elif opcode == 0b011:
        regs[7] = keep_16bits(pc + 1)
        pc = imm
    return valid_pc(pc), regs, memory, cache


#given a 7 bit number, do the sign extend
def sign_extend_7(num):
    left_most_bit = (num & 64) >> 6
    if left_most_bit == 0b1:
        return 65408 | num
    else:
        return num


#produce sign number, which can be positive or negative
def sign_number_converter(num, bits):
    left_most_bit = (num & 64) >> 6
    if left_most_bit == 0b1:
        bitmask = (1 << bits) - 1
        return -((num ^ bitmask) + 1)
    else:
        return num


#keep the memory address to be valid
#make sure the address is inside the range of valid address
def valid_pc(pc):
    return pc % 65535


#keep the number to be 16 bits number
def keep_16bits(num):
    bitmask = (1 << 16) - 1
    return num & bitmask


# when lw instruction has been called
# load_data will check whether the value of given address
# has been stored in a cache or caches and where it has been
# store in the cache and update LRU
def load_data(cache, addr, pc):
    if len(cache) == 4:
        [L1, L1size, L1assoc, L1blocksize] = [x for x in cache]
        row1, tag1 = row_tag_calculator(addr, L1size, L1assoc, L1blocksize)
        status = status_check(L1, row1, tag1)
        # update cache
        L1 = LRU(L1, row1, tag1)
        print_log_entry("L1", status, pc, addr, row1)
        return [L1, L1size, L1assoc, L1blocksize]
    else:
        [L1, L1size, L1assoc, L1blocksize,
         L2, L2size, L2assoc, L2blocksize] = [x for x in cache]
        row1, tag1 = row_tag_calculator(addr, L1size, L1assoc, L1blocksize)
        status = status_check(L1, row1, tag1)
        print_log_entry("L1", status, pc, addr, row1)
        L1 = LRU(L1, row1, tag1)

        if status == "MISS":
            row2, tag2 = row_tag_calculator(addr, L2size, L2assoc, L2blocksize)
            status = status_check(L2, row2, tag2)
            print_log_entry("L2", status, pc, addr, row2)
            L2 = LRU(L2, row2, tag2)
        return [L1, L1size, L1assoc, L1blocksize,
         L2, L2size, L2assoc, L2blocksize]


# when sw called, stored the data into cache,
# if there is more than one cache, we also
# need to store the value into other cache with right row and tag.
def store_data(cache, pc, addr):
    if len(cache) == 4:
        [L1, L1size, L1assoc, L1blocksize] = [x for x in cache]
        row1, tag1 = row_tag_calculator(addr, L1size, L1assoc, L1blocksize)
        L1 = LRU(L1, row1, tag1)
        print_log_entry("L1", "SW", pc, addr, row1)
        return [L1, L1size, L1assoc, L1blocksize]
    else:
        [L1, L1size, L1assoc, L1blocksize,
         L2, L2size, L2assoc, L2blocksize] = [x for x in cache]
        row1, tag1 = row_tag_calculator(addr, L1size, L1assoc, L1blocksize)
        print_log_entry("L1", "SW", pc, addr, row1)
        L1 = LRU(L1, row1, tag1)

        row2, tag2 = row_tag_calculator(addr, L2size, L2assoc, L2blocksize)
        print_log_entry("L2", "SW", pc, addr, row2)
        L2 = LRU(L2, row2, tag2)

        return [L1, L1size, L1assoc, L1blocksize,
         L2, L2size, L2assoc, L2blocksize]


# keep track of the order in which blocks have been used
def LRU(cache, row, tag):
    if tag in cache[row]:
        index = cache[row].index(tag)
        for i in range(index, len(cache[row]) - 1):
            cache[row][i] = cache[row][i + 1]
    else:
        for i in range(len(cache[row]) - 1):
            cache[row][i] = cache[row][i + 1]
    cache_row = cache[row].copy()
    cache_row[-1] = tag
    cache[row] = cache_row
    return cache


# calculate the responding row and tag of given address
def row_tag_calculator(addr, cache_size, associativity, blocksize):
    cache_row = cache_size // (associativity * blocksize)
    blockid = addr // blocksize
    row = blockid % cache_row
    tag = blockid // cache_row
    return row,tag


# check hit or miss
def status_check(cache, row, tag):
    if tag in cache[row]:
        status = "HIT"
    else:
        status = "MISS"
    return status


def main():
    parser = argparse.ArgumentParser(description='Simulate E20 cache')
    parser.add_argument('filename', help=
    'The file containing machine code, typically with .bin suffix')
    parser.add_argument('--cache', help=
    'Cache configuration: size,associativity,blocksize (for one cache) '
    'or size,associativity,blocksize,size,associativity,blocksize (for two caches)')
    cmdline = parser.parse_args()

    # initialize program counter, value of registers, and value of all memory
    pc = 0
    regs = [0] * constants.NUM_REGS
    memory = [0] * constants.MEM_SIZE

    with open(cmdline.filename) as file:
        #pass # TODO: your code here. Load file and parse using load_machine_code
        load_machine_code(file, memory)

    if cmdline.cache is not None:
        parts = cmdline.cache.split(",")
        if len(parts) == 3:
            [L1size, L1assoc, L1blocksize] = [int(x) for x in parts]
            L1rows = L1size // (L1assoc * L1blocksize)
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1rows)
            # initialize: L1 cache
            L1 = [[None] * L1assoc] * L1rows
            cache = [L1, L1size, L1assoc, L1blocksize]
        elif len(parts) == 6:
            [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize] = \
                [int(x) for x in parts]
            L1rows = L1size // (L1assoc * L1blocksize)
            print_cache_config("L1", L1size, L1assoc, L1blocksize, L1rows)
            # initialize: L1 cache
            L1 = [[None] * L1assoc] * L1rows
            L2rows = L2size // (L2assoc * L2blocksize)
            print_cache_config("L2", L2size, L2assoc, L2blocksize, L2rows)
            # initialize: L2 cache
            L2 = [[None] * L2assoc] * L2rows
            cache = [L1, L1size, L1assoc, L1blocksize, L2, L2size, L2assoc, L2blocksize]
        else:
            raise Exception("Invalid cache config")


    # TODO: your code here. Do simulation.
    old_pc = pc
    while True:
        new_pc, new_regs, new_memory, new_cache = simulation(old_pc, regs, memory, cache)
        if new_pc == old_pc:
            if memory[old_pc % 8191] >> 13 == 0b010:  # j indicate halt
                break
        # update pc, reg and memory
        old_pc = new_pc
        regs = new_regs
        memory = new_memory
        cache = new_cache


if __name__ == "__main__":
    main()
