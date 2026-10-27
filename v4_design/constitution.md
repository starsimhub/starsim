# Constitution of v4

[*Authorship:* 🧠 99%, 🤖 1%](# "Humans wrote everything except 'tidy' and 'for whom' were suggested by Claude")

This document provides the foundational guidance for how v4 should be built. Primary principles get the most weight, in order, followed by secondary principles, also in order; development principles are more about how work is done than the final outcome. However, use your judgement: e.g. simplicity comes above maintainability, but a small improvement in simplicity does not justify a big loss of maintainability.

## Primary principles

1. **Simplicity:** It should be easy and intuitive to write and modify disease models in the supported forms (compartmental, stochastic compartmental, metapopulation, ABM). "Simple" means for users who are epidemiologists, not engineers. Code ported from v3 to v4 should become simpler, as should (most) models from other frameworks ported into v4.
2. **Performance:** The models should be as fast as possible; as close as is feasible to the speed of custom compiled code.
3. **Flexibility:** All major disease modeling features should be expressible in v4 (which is not to say every feature has to be *implemented*, just that it has to be possible and intuitive to implement).
4. **Correctness:** The machinery should be correct and validated. However, (over-)correctness should not come at a cost to simplicity or performance, e.g. Euler steps, float32, and other approximations are (usually) fine.

## Secondary principles

1. **Understandability:** Avoid having hundreds of concepts users need to learn, or complex and brittle subsystems. However, this should not come at a (big) cost to flexibility.
2. **Maintainability:** Limit duplication and interdependencies, e.g. multiple solvers reimplementing the same thing that can drift apart. However, some duplication may be necessary to achieve high performance.
3. **Composability:** Users should be able to use as much or as little of v4 as they want; for example using only its integration loop with custom modules (ODE/ABM, in NumPy or otherwise). This includes within a module: declared and coded parts should coexist peacefully, so if part of a module is hard to declare, ideally only that part needs to be code.
4. **Startup time:** Running a simple model should not take a long pre-compilation step. However, explicit compilation is an option for big runs (e.g. calibration). One-time compilation is more acceptable, but be careful about environments that don't support caching, and should be opt-in rather than required.
5. **AI-friendliness:** Features of v4 (and their documentation) should be written in a way that AI tools find them easy to work with. However, this should not come at a cost to human-readability and simplicity.

## Development principles

1. **Work nimbly:** When making large library updates, focus on only the code and the relevant tests. Don't change every doc, import script, and test for every rename. Instead mark what's stale.
2. **Think big picture:** Always ask: "How can the whole system be better?". For example, don't make a new custom pathway just to improve performance for one very specific application, if it reduces understandability, maintainability, and simplicity.
3. **Think outside the box:** Don't take *anything* as an invariant. Numba indexing is slow? We can fix Numba. Sciris function missing a feature? We can add to Sciris. STIsim needs a shim? We can rewrite STIsim. (But write up proposed fixes to other libraries; do not just go ahead and do them.)
4. **No baggage:** While we should avoid needless backwards incompatibilities (e.g., renaming classes that are unchanged), there is also no need to leave in legacy compatibility features: migration scripts and AI skills can handle porting from v3 to v4.
5. **No bloat:** Try to avoid bloat, including lines of code, but don't enforce rigid limits; satisfying many of the other principles will obviously require new lines of code. But try to minimize multiple ways of doing the same thing, unnecessary tests, complex code for corner cases, etc.
6. **Tidy as you go:** After each phase of work, look for things to delete/simplify. Don't keep something just because it's "in the spec"; ask if the spec can be simplified.
7. **Ask "for whom":** A new class, function, keyword, or abstraction needs a real user: its own tests, docs, and spec don't count. If nothing needs it, use the simplest, most common built-in/plain alternative. Don't write tests that enumerate classes or keep docs in sync with code when both could be removed instead.