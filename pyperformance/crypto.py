#!/usr/bin/env python
"""
Pure-Python Implementation of the AES block-cipher.

Benchmark AES in CTR mode using the pyaes module.
"""

import pyaes
import timeit

from pyperformance.utils import run_benchmark

# 23,000 bytes
CLEARTEXT = b"This is a test. What could possibly go wrong? " * 500

# 128-bit key (16 bytes)
KEY = b"\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,"


def aes_encrypt_decrypt(key, cleartext):
    aes = pyaes.AESModeOfOperationCTR(key)
    ciphertext = aes.encrypt(cleartext)

    # need to reset IV for decryption
    aes = pyaes.AESModeOfOperationCTR(key)
    plaintext = aes.decrypt(ciphertext)

    # explicitly destroy the pyaes object
    aes = None

    if plaintext != cleartext:
        raise Exception("decrypt error!")


def bench_pyaes(loops, key=KEY, cleartext=CLEARTEXT):
    def run_aes():
        aes_encrypt_decrypt(key, cleartext)

    return timeit.timeit(run_aes, number=loops)


BENCHMARKS = {"aes": (bench_pyaes, KEY, CLEARTEXT)}


if __name__ == "__main__":
    for bench_name in sorted(BENCHMARKS):
        run_benchmark(bench_name, BENCHMARKS, 20)
