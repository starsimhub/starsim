"""
Rust backend for Starsim (``import starsim.rust as ssr``).

This subpackage is the home of the Python->Rust effort. The goal: users build
and think in Python, and opt into Rust-accelerated modules where it pays off,
with a validation harness guaranteeing the Rust path matches the Python one.

Status: **Phase 1 (in progress).** The equivalence-validation harness
(:func:`compare`) is in place, and ``ssr.SIS`` is Rust-accelerated for its
transmission kernel (``compute_transmission``), validated byte-identical against
``ss.SIS``. ``ssr.RandomNet`` is a not-yet-accelerated pass-through. Modules fall
back to pure Python if the compiled kernel (``starsim_rust_kernels``) is absent;
check :data:`starsim.rust.modules.available`.

Design notes live in ``SUPPORTED_SUBSET.md`` (what a module's ``step()`` may
contain to be portable) alongside this file. To port a new module, use the
starsim-rust-port skill; to check portability, use the rust-portability-linter
agent.
"""
from .validate import compare, ValidationReport, TIERS
from .modules import SIS, SIR, RandomNet, Births, Deaths, run_engine, all_native, available

__all__ = ['compare', 'ValidationReport', 'TIERS', 'SIS', 'SIR', 'RandomNet',
           'Births', 'Deaths', 'run_engine', 'all_native', 'available']
