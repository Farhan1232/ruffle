use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::function::FunctionArgs;
pub use crate::avm2::object::dictionary_allocator;
use crate::avm2::parameters::ParametersExt;
use crate::avm2::value::Value;

/// Implements `Dictionary.init`, the constructor's half that records
/// `weakKeys`.
pub fn init<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    let this = this.as_object().unwrap();
    let this = this.as_dictionary_object().unwrap();

    if args.get_bool(0) {
        this.set_weak_keys(activation);
    }

    Ok(Value::Undefined)
}
