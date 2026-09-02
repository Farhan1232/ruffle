//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::{DynamicKey, DynamicMap};
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject, WeakObject};
use crate::avm2::value::Value;
use crate::library::Resurrector;
use crate::string::AvmString;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Finalization, Gc, GcWeak, Mutation, RefLock};
use ruffle_common::utils::HasPrefixField;
use std::cell::Cell;
use std::hash::{Hash, Hasher};

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
            weak_entries: RefLock::new(DynamicMap::new()),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,

    /// Whether this dictionary was constructed with `weakKeys`.
    #[collect(require_static)]
    weak_keys: Cell<bool>,

    /// The object-keyed entries of a weak-keyed dictionary. Empty and unused
    /// for a dictionary with strong keys, whose object keys live in the base
    /// object's property map like any other property.
    ///
    /// Keys are held weakly, and values are deliberately hidden from the
    /// collector while it marks: an entry keeps its value alive only for as
    /// long as its key is reachable from somewhere *other* than this table.
    /// See [`DictionaryObject::resurrect_live_entries`] for how that decision
    /// is made and [`crate::avm2::Avm2::weak_dictionaries`] for who asks.
    weak_entries: RefLock<DynamicMap<WeakKey<'gc>, WeakValue<'gc>>>,
}

/// A dictionary key held weakly and compared by identity.
#[derive(Clone, Copy, Collect)]
#[collect(no_drop)]
struct WeakKey<'gc>(WeakObject<'gc>);

impl PartialEq for WeakKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl Eq for WeakKey<'_> {}

impl Hash for WeakKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0.as_ptr() as usize).hash(state);
    }
}

impl<'gc> WeakKey<'gc> {
    fn is_dead(self, fc: &Finalization<'gc>) -> bool {
        // A key that has already been dropped cannot be upgraded; one that is
        // merely unreached in this cycle can, and `is_dead` says which.
        self.0
            .upgrade(fc)
            .is_none_or(|object| Gc::is_dead(fc, object.gc_base()))
    }
}

/// The value of a weak-keyed entry.
///
/// Not traced during marking - see [`DictionaryObjectData::weak_entries`] -
/// so a value that nothing else references stays white until the finalization
/// pass either resurrects it (its key is alive) or drops the entry (its key is
/// dead). Either happens before anything is swept, so the pointer is never
/// left dangling.
#[derive(Clone)]
struct WeakValue<'gc> {
    value: Value<'gc>,
    /// Set by the finalization pass once this cycle has resurrected the
    /// value, so that a later round of the same pass does not report it as
    /// new work; cleared when the pass finishes.
    kept: Cell<bool>,
}

unsafe impl<'gc> Collect<'gc> for WeakValue<'gc> {
    const NEEDS_TRACE: bool = false;

    fn trace<C: Trace<'gc>>(&self, _cc: &mut C) {}
}

/// Enumeration indices at or above this refer to the weak entries; below it,
/// to the base object's properties. `for..in` only ever hands the index back
/// to the object that produced it.
const WEAK_INDEX_BASE: u32 = 1 << 30;

impl<'gc> DictionaryObject<'gc> {
    /// Whether object keys are held weakly.
    pub fn weak_keys(self) -> bool {
        self.0.weak_keys.get()
    }

    /// Switches this dictionary to weak keys, and registers it with the
    /// collector. Called from the constructor, before any entry exists.
    pub fn set_weak_keys(self, activation: &mut Activation<'_, 'gc>) {
        if !self.0.weak_keys.replace(true) {
            activation
                .avm2()
                .register_weak_dictionary(DictionaryObjectWeak(Gc::downgrade(self.0)));
        }
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        if self.weak_keys() {
            return self
                .0
                .weak_entries
                .borrow()
                .get(&WeakKey(name.downgrade()))
                .map(|v| v.value.value)
                .unwrap_or(Value::Undefined);
        }
        self.base()
            .values()
            .get(&DynamicKey::Object(name))
            .map(|v| v.value)
            .unwrap_or(Value::Undefined)
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        if self.weak_keys() {
            unlock!(Gc::write(mc, self.0), DictionaryObjectData, weak_entries)
                .borrow_mut()
                .insert(
                    WeakKey(name.downgrade()),
                    WeakValue {
                        value,
                        kept: Cell::new(false),
                    },
                );
            return;
        }
        self.base()
            .values_mut(mc)
            .insert(DynamicKey::Object(name), value);
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        if self.weak_keys() {
            unlock!(Gc::write(mc, self.0), DictionaryObjectData, weak_entries)
                .borrow_mut()
                .remove(&WeakKey(name.downgrade()));
            return;
        }
        self.base().values_mut(mc).remove(&DynamicKey::Object(name));
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        if self.weak_keys() {
            return self
                .0
                .weak_entries
                .borrow()
                .contains_key(&WeakKey(name.downgrade()));
        }
        self.base().values().contains_key(&DynamicKey::Object(name))
    }

    /// Number of weakly-keyed entries currently in the table. Only for
    /// reporting; a key may be unreachable and still counted until the next
    /// collection cycle prunes it.
    pub fn weak_entry_count(self) -> usize {
        self.0.weak_entries.borrow().len()
    }

    /// The finalization step for a weak-keyed dictionary: every entry whose
    /// key was reached by marking has its value brought back, so that the
    /// value lives exactly as long as the key does. Returns whether anything
    /// was resurrected this round; if so, marking has to resume - the value
    /// may reach other weak keys - and this is asked again.
    ///
    /// Only meaningful during finalization, once marking has finished.
    pub(crate) fn resurrect_live_entries(self, fc: &Finalization<'gc>) -> bool {
        let mut resurrected = false;
        for (key, entry) in self.0.weak_entries.borrow().iter() {
            let value = &entry.value;
            if value.kept.get() || key.is_dead(fc) {
                continue;
            }
            value.kept.set(true);
            value.value.trace(&mut Resurrector(fc));
            resurrected = true;
        }
        resurrected
    }

    /// Drops every entry whose key is dead, once the finalization pass has
    /// reached its fixpoint, and clears the marks of the ones it kept. Must
    /// run before the sweep: the values of dropped entries were never traced.
    pub(crate) fn prune_dead_entries(self, fc: &Finalization<'gc>) {
        let dead: Vec<WeakKey<'gc>> = self
            .0
            .weak_entries
            .borrow()
            .iter()
            .filter_map(|(key, entry)| {
                if key.is_dead(fc) {
                    Some(*key)
                } else {
                    entry.value.kept.set(false);
                    None
                }
            })
            .collect();
        if !dead.is_empty() {
            let mut entries =
                unlock!(Gc::write(fc, self.0), DictionaryObjectData, weak_entries).borrow_mut();
            for key in dead {
                entries.remove(&key);
            }
        }
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_next_enumerant(
        self,
        last_index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<u32, Error<'gc>> {
        if !self.weak_keys() {
            return Ok(self.base().get_next_enumerant(last_index));
        }
        // The base object's properties first, then the weak entries.
        if last_index < WEAK_INDEX_BASE {
            let next = self.base().get_next_enumerant(last_index);
            if next != 0 {
                return Ok(next);
            }
            return Ok(self.next_weak_enumerant(0));
        }
        Ok(self.next_weak_enumerant(last_index - WEAK_INDEX_BASE))
    }

    fn get_enumerant_name(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if self.weak_keys() && index >= WEAK_INDEX_BASE {
            return Ok(self
                .0
                .weak_entries
                .borrow()
                .key_at((index - WEAK_INDEX_BASE) as usize)
                .and_then(|key| key.0.upgrade(activation.gc()))
                .map(Value::Object)
                .unwrap_or(Value::Null));
        }
        Ok(self.base().get_enumerant_name(index).unwrap_or(Value::Null))
    }

    fn get_enumerant_value(
        self,
        index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if self.weak_keys() && index >= WEAK_INDEX_BASE {
            return Ok(self
                .0
                .weak_entries
                .borrow()
                .value_at((index - WEAK_INDEX_BASE) as usize)
                .map(|v| v.value)
                .unwrap_or(Value::Undefined));
        }
        Ok(*self
            .base()
            .values()
            .value_at(index as usize)
            .unwrap_or(&Value::Undefined))
    }
}

impl<'gc> DictionaryObject<'gc> {
    fn next_weak_enumerant(self, last_weak_index: u32) -> u32 {
        self.0
            .weak_entries
            .borrow()
            .next(last_weak_index as usize)
            .map(|index| index as u32 + WEAK_INDEX_BASE)
            .unwrap_or(0)
    }
}
