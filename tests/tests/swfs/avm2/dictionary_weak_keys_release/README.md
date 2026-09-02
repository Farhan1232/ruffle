# dictionary_weak_keys_release

`Test.as` fills a weak-keyed `Dictionary` with twenty entries whose keys are
referenced from nowhere but the dictionary itself - and, through each entry's
*value*, from the entry - alongside a key the movie keeps, a string key and an
integer key. A strong-keyed dictionary gets the same twenty entries as a
control.

The Rust side (`tests/weak_dictionary/mod.rs`) forces two full collections at
frame 20. At frame 30 the movie reports what survived: the twenty unreferenced
entries must be gone from the weak dictionary and still present in the strong
one, and the held, string and integer keys must still resolve, enumerate and
delete.

Rebuild `test.swf` with `build.py <playerglobal_import.abc>`, using the
`asc.jar` in `tools/asc`.
