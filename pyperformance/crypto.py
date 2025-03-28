#!/usr/bin/env python
"""
Pure-Python Implementation of the AES block-cipher.

Benchmark AES in CTR mode using the pyaes module.
"""

import pyaes
import pyperf
import numba


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


@numba.jit(nopython=False, forceobj=True)
def aes_encrypt_decrypt_numba(key, cleartext):
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
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        aes_encrypt_decrypt(key, cleartext)

    return pyperf.perf_counter() - t0


def bench_pyaes_numba(loops, key=KEY, cleartext=CLEARTEXT):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        aes_encrypt_decrypt_numba(key, cleartext)

    return pyperf.perf_counter() - t0


BENCHMARKS = {
    "aes": (bench_pyaes, KEY, CLEARTEXT),
    # "aes_numba": (bench_pyaes_numba, KEY, CLEARTEXT),
}


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.argparser.set_defaults(quiet=False)
    for bench in sorted(BENCHMARKS):
        name = "crypto_%s" % bench
        args = BENCHMARKS[bench]
        runner.bench_time_func(name, *args)
