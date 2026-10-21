# Constitution of v4

This document provides the foundational guidance for how v4 should be built.

## Primary principles

1. **Simplicity:** It should be easy and intuitive to write and modify disease models in the supported forms (ODE, SDE, ABM, metapopulation).
2. **Performance:** The models should be as fast as possible; as close as is feasible to the speed of custom compiled code (e.g. Rust).
3. **Flexibility:** All major disease modeling features should be expressible in v4 (which is not to say every feature has to be *implemented*, just that it has to be possible and intuitive to implement).

## Secondary principles

1. **Understandability:** Avoid having hundreds of concepts users need to learn, or complex and brittle subsystems. However, this should not come at a (big) cost to flexibility.
2. **Maintainability:** Limit duplication and interdependencies, e.g. multiple solvers reimplementing the same thing that can drift apart. However, some duplication may be necessary to achieve high performance.
3. **Composability:** Users should be able to use as much or as little of v4 as they want; for example using only its integration loop with custom modules (ODE/ABM, in NumPy or otherwise), or using v4 modules with a custom integration loop.
4. **Startup time:** Running a simple model should not take a long pre-compilation step. However, explicit compilation is an option for big runs (e.g. calibration). One-time compilation is more acceptable, but be careful about environments that don't support caching, and should be opt-in rather than required.
5. **AI-friendliness:** Features of v4 (and their documentation) should be written in a way that AI tools find them easy to work with. However, this should not come at a cost to human-readability and simplicity.

## Development principles

1. **Work nimbly:** When making large library updates, focus on only the code and the relevant tests. Don't change every doc, import script, and test for every rename. Instead mark what's stale.
2. **Think big picture:** Always ask: "How can the whole system be better?". For example, don't make a new custom pathway just to improve performance for one very specific application, if it reduces understandability, maintainability, and simplicity.
3. **Think outside the box:** Don't take *anything* as an invariant. Numba indexing is slow? We can fix Numba. Sciris function missing a feature? We can add to Sciris. STIsim needs a shim? We can rewrite STIsim.
4. **Avoid bloat:** Try to avoid bloat, including lines of code, but don't enforce rigid limits; satisfying many of the other principles will obviously require new lines of code. But try to minimize multiple ways of doing the same thing, unnecessary tests, complex code for corner cases, etc.